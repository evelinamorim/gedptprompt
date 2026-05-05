"""
run_inference.py
Query an Ollama-hosted SLM for Grammar Error Detection on Brazilian Portuguese.

For each sentence in the test split, the model is asked to label each token
with O, B-WRONG, or I-WRONG and the result is saved to a JSON file.

Usage:
    python run_inference.py --config config.yaml --split test
    python run_inference.py --config config.yaml --split val --strategy few_shot
"""

from __future__ import annotations
import argparse
import json
import re
import time
from pathlib import Path
import os

import requests
import yaml

from data_reader import Sentence, read_bio_file


# ------------------------------------------------------------------ #
# Prompt templates
# ------------------------------------------------------------------ #

SYSTEM_PROMPT = """\
You are a grammar error detection system for Brazilian Portuguese.
Your task is to label each token in a sentence with one of three tags:
  - O          : the token is grammatically correct
  - B-WRONG    : the token is the BEGINNING of a grammatical error span
  - I-WRONG    : the token is INSIDE (continuing) a grammatical error span

Rules:
1. Every token in the input must receive exactly one label.
2. An error span always starts with B-WRONG and may be followed by one or more I-WRONG tokens.
3. Do NOT output any explanation, commentary, or extra text.
4. Output ONLY a valid JSON object with this exact structure:
   {"labels": ["O", "B-WRONG", "I-WRONG", ...]}
   The "labels" list must have the SAME number of elements as the input token list.
"""

ZERO_SHOT_USER_TEMPLATE = """\
Tokens: {tokens_json}

Respond with a JSON object: {{"labels": [...]}}
"""

FEW_SHOT_EXAMPLE_TEMPLATE = """\
Example {n}:
Tokens: {tokens_json}
Output: {{"labels": {labels_json}}}
"""

FEW_SHOT_USER_TEMPLATE = """\
{examples}

Now label the following tokens:
Tokens: {tokens_json}

Respond with a JSON object: {{"labels": [...]}}
"""


# ------------------------------------------------------------------ #
# Prompt builders
# ------------------------------------------------------------------ #

def build_zero_shot_prompt(sentence: Sentence) -> str:
    tokens_json = json.dumps(sentence.tokens, ensure_ascii=False)
    return ZERO_SHOT_USER_TEMPLATE.format(tokens_json=tokens_json)


def build_few_shot_prompt(sentence: Sentence, examples: list[Sentence]) -> str:
    example_blocks = []
    for i, ex in enumerate(examples, 1):
        block = FEW_SHOT_EXAMPLE_TEMPLATE.format(
            n=i,
            tokens_json=json.dumps(ex.tokens, ensure_ascii=False),
            labels_json=json.dumps(ex.labels, ensure_ascii=False),
        )
        example_blocks.append(block)
    examples_str = "\n".join(example_blocks)
    tokens_json = json.dumps(sentence.tokens, ensure_ascii=False)
    return FEW_SHOT_USER_TEMPLATE.format(
        examples=examples_str,
        tokens_json=tokens_json,
    )


# ------------------------------------------------------------------ #
# Few-shot example selection
# ------------------------------------------------------------------ #

def select_few_shot_examples(
    train_sentences: list[Sentence],
    n: int,
    *,
    prefer_errors: bool = True,
) -> list[Sentence]:
    """
    Select n few-shot examples from training data.
    By default tries to pick examples that contain at least one error span,
    to help the model understand the error labeling task.
    """
    if prefer_errors:
        with_errors = [s for s in train_sentences if any(l != "O" for l in s.labels)]
        pool = with_errors if len(with_errors) >= n else train_sentences
    else:
        pool = train_sentences
    # deterministic: pick every k-th sentence to spread across the file
    step = max(1, len(pool) // n)
    return [pool[i * step] for i in range(n)]


# ------------------------------------------------------------------ #
# Ollama API call
# ------------------------------------------------------------------ #

def query_ollama(
    user_prompt: str,
    *,
    host: str,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    num_ctx: int,
    retries: int = 3,
    backoff: float = 2.0,
) -> str:
    """Send a chat completion request to Ollama and return the assistant text."""
    url = f"{host.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "format": {                          # <-- add this block
        "type": "object",
        "properties": {
            "labels": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["O", "B-WRONG", "I-WRONG"]
                }
            }
          },
          "required": ["labels"]
        },
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": max_tokens,
            "num_ctx": num_ctx,
        },
    }

    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"]
        except (requests.RequestException, KeyError) as exc:
            if attempt == retries:
                raise
            wait = backoff ** attempt
            print(f"  [retry {attempt}/{retries}] error: {exc}. Waiting {wait:.1f}s …")
            time.sleep(wait)
    return ""  # unreachable


# ------------------------------------------------------------------ #
# Response parsing
# ------------------------------------------------------------------ #

_JSON_RE = re.compile(r'\{.*?"labels"\s*:\s*\[.*?\]\s*\}', re.DOTALL)

def parse_labels(response_text: str, expected_len: int, valid_labels: set[str]) -> list[str]:
    """
    Extract the labels list from the model response.
    Falls back to all-O if parsing fails or lengths mismatch.
    """
    match = _JSON_RE.search(response_text)
    if match:
        try:
            obj = json.loads(match.group())
            labels = obj.get("labels", [])
            # Sanitise: replace unknowns with O
            labels = [l if l in valid_labels else "O" for l in labels]
            if len(labels) == expected_len:
                return labels
            # Length mismatch: pad or truncate
            if len(labels) < expected_len:
                labels += ["O"] * (expected_len - len(labels))
            else:
                labels = labels[:expected_len]
            return labels
        except json.JSONDecodeError:
            pass

    print(f"  [WARNING] Could not parse JSON from response. Defaulting to all-O.")
    return ["O"] * expected_len


# ------------------------------------------------------------------ #
# Main inference loop
# ------------------------------------------------------------------ #

def run_inference(cfg: dict, split: str, strategy: str) -> None:
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    prompt_cfg = cfg["prompting"]
    out_cfg = cfg["output"]
    label_cfg = cfg["labels"]

    valid_labels = set(label_cfg["valid_set"])
    model_name = model_cfg["name"]
    model_tag = model_name.replace(":", "-").replace("/", "-")

    # Load data
    test_path = data_cfg[f"{split}_file"]
    print(f"Loading {split} split from: {test_path}")
    sentences = read_bio_file(test_path, valid_labels=valid_labels)
    print(f"  {len(sentences)} sentences loaded.")

    # Load few-shot examples if needed
    few_shot_examples: list[Sentence] = []
    if strategy == "few_shot":
        train_path = prompt_cfg["few_shot_source"]
        print(f"Loading few-shot examples from: {train_path}")
        train_sents = read_bio_file(train_path, valid_labels=valid_labels)
        few_shot_examples = select_few_shot_examples(
            train_sents, prompt_cfg["num_few_shot_examples"]
        )
        print(f"  Selected {len(few_shot_examples)} examples.")

    # Prepare output path
    pred_path = out_cfg["predictions_file"].format(
        model_tag=model_tag, split=split, strategy=strategy
    )
    Path(pred_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Resume from partial if it exists
    partial_path = pred_path + ".partial"
    results: list[dict] = []
    completed_ids: set[int] = set()

    if Path(partial_path).exists():
        print(f"Found partial predictions at: {partial_path}")
        with open(partial_path, encoding="utf-8") as fh:
            partial_data = json.load(fh)
        results = partial_data["predictions"]
        completed_ids = {r["sentence_id"] for r in results}
        print(f"  Resuming from sentence {max(completed_ids) + 1} ({len(completed_ids)} already done)")

    # Filter out already completed sentences
    sentences = [s for s in sentences if s.id not in completed_ids]
    n = len(sentences)
    print(f"  {n} sentences remaining.")

    for i, sent in enumerate(sentences, 1):
        if i % 50 == 0 or i == 1:
            print(f"  [{i}/{n}] sentence id={sent.id} …")

        # Build prompt
        if strategy == "few_shot":
            user_prompt = build_few_shot_prompt(sent, few_shot_examples)
        else:
            user_prompt = build_zero_shot_prompt(sent)

        # Query model
        raw_response = query_ollama(
            user_prompt,
            host=model_cfg["ollama_host"],
            model=model_name,
            temperature=model_cfg["temperature"],
            top_p=model_cfg["top_p"],
            max_tokens=model_cfg["max_tokens"],
            num_ctx=model_cfg["num_ctx"],
        )

        # Parse labels
        pred_labels = parse_labels(raw_response, len(sent.tokens), valid_labels)

        results.append({
            "sentence_id": sent.id,
            "tokens": sent.tokens,
            "gold_labels": sent.labels,
            "pred_labels": pred_labels,
            "raw_response": raw_response,
        })
        if i % 100 == 0:
            with open(pred_path + ".partial", "w", encoding="utf-8") as fh:
                json.dump({
                    "model": model_name,
                    "split": split,
                    "strategy": strategy,
                    "num_sentences": len(results),
                    "predictions": results,
                }, fh, ensure_ascii=False, indent=2)
            print(f"  [checkpoint] {len(results)} sentences saved.")

    partial_path = pred_path + ".partial"
    with open(partial_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "model": model_name,
                "split": split,
                "strategy": strategy,
                "num_sentences": len(results),
                "predictions": results,
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )
    os.rename(partial_path, pred_path)
    print(f"\nPredictions saved to: {pred_path}")

# ------------------------------------------------------------------ #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GED-PT SLM inference via Ollama")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--split", default="test", choices=["test", "val"], help="Dataset split to run"
    )
    parser.add_argument(
        "--strategy",
        default=None,
        help="Prompting strategy: zero_shot or few_shot (overrides config)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    strategy = args.strategy or cfg["prompting"]["strategy"]
    run_inference(cfg, args.split, strategy)
