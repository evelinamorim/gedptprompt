"""
run_inference.py
Grammar Error Detection for Brazilian Portuguese using HuggingFace Transformers.
Replaces the previous Ollama-based implementation for faster batched inference.

Usage:
    python run_inference.py --config config.yaml --split test
    python run_inference.py --config config.yaml --split test --strategy few_shot
    python run_inference.py --config config.yaml --split test --model Qwen/Qwen3-8B
"""

from __future__ import annotations
import argparse
import json
import os
import re
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    max_span_length: int = 5,
    max_sentence_length: int = 30,
    contrastive: bool = False,
) -> list[Sentence]:

    if contrastive:
        correct = [
            s for s in train_sentences
            if all(l == "O" for l in s.labels)
            and len(s) <= max_sentence_length
        ]
        with_errors = [
            s for s in train_sentences
            if any(l != "O" for l in s.labels)
            and all(end - start <= max_span_length for start, end in s.error_spans())
            and len(s) <= max_sentence_length
        ]
        correct_pick = [correct[len(correct) // 2]] if correct else []
        step = max(1, len(with_errors) // (n - 1))
        error_picks = [with_errors[i * step] for i in range(n - 1)]
        return correct_pick + error_picks

    with_errors = [
        s for s in train_sentences
        if any(l != "O" for l in s.labels)
        and all(end - start <= max_span_length for start, end in s.error_spans())
        and len(s) <= max_sentence_length
    ]
    pool = with_errors if len(with_errors) >= n else train_sentences
    step = max(1, len(pool) // n)
    return [pool[i * step] for i in range(n)]


# ------------------------------------------------------------------ #
# Model loading
# ------------------------------------------------------------------ #

def load_model(model_id: str, hf_cache: str) -> tuple:
    """Load tokenizer and model onto GPU."""
    print(f"Loading model: {model_id}")
    print(f"  HF cache: {hf_cache}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=hf_cache,
        trust_remote_code=True,
        padding_side="left",
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=hf_cache,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Model loaded. Parameters: {n_params:.2f}B")
    return tokenizer, model


# ------------------------------------------------------------------ #
# Dynamic batch sizing
# ------------------------------------------------------------------ #

def estimate_batch_size(model: AutoModelForCausalLM) -> int:
    if not torch.cuda.is_available():
        return 1
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    model_params_b = sum(p.numel() for p in model.parameters()) / 1e9
    model_vram = model_params_b * 2   # bf16 = 2 bytes/param
    available = total_vram - model_vram - 2.0
    batch_size = max(1, min(int(available / 1.0), 32))
    print(f"  Dynamic batch size: {batch_size} "
          f"(VRAM: {total_vram:.1f}GB, model: {model_vram:.1f}GB)")
    return batch_size


# ------------------------------------------------------------------ #
# Chat prompt formatting
# ------------------------------------------------------------------ #

def format_chat_prompt(user: str, tokenizer: AutoTokenizer, model_id: str) -> str:
    """Format system+user prompt using the model's chat template."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    try:
        # disable thinking for Qwen3 and Gemma4
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # Fallback for models without chat template (e.g. Tucano base)
        prompt = (
            f"<|system|>\n{SYSTEM_PROMPT}\n"
            f"<|user|>\n{user}\n"
            f"<|assistant|>\n"
        )
    return prompt


# ------------------------------------------------------------------ #
# Batch generation
# ------------------------------------------------------------------ #

def generate_batch(
    prompts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    max_new_tokens: int = 512,
) -> list[str]:
    """Run a batch of prompts and return generated text per prompt."""
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    responses = []
    input_len = inputs["input_ids"].shape[1]
    for output in outputs:
        new_tokens = output[input_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        responses.append(text)
    return responses


# ------------------------------------------------------------------ #
# Response parsing
# ------------------------------------------------------------------ #

_JSON_RE = re.compile(r'\{.*?"labels"\s*:\s*\[.*?\]\s*\}', re.DOTALL)
VALID_LABELS = {"O", "B-WRONG", "I-WRONG"}


def parse_labels(response_text: str, expected_len: int) -> list[str]:
    """Extract labels from model response with fallback for truncated JSON."""
    # Strip thinking blocks (Qwen3, DeepSeek)
    response_text = re.sub(
        r'<think>.*?</think>', '', response_text, flags=re.DOTALL
    ).strip()

    # Try full JSON parse
    match = _JSON_RE.search(response_text)
    if match:
        try:
            obj = json.loads(match.group())
            labels = obj.get("labels", [])
            labels = [l if l in VALID_LABELS else "O" for l in labels]
            if len(labels) == expected_len:
                return labels
            if len(labels) < expected_len:
                labels += ["O"] * (expected_len - len(labels))
            else:
                labels = labels[:expected_len]
            return labels
        except json.JSONDecodeError:
            pass

    # Fallback: extract individual label strings from truncated response
    partial = re.findall(r'"(O|B-WRONG|I-WRONG)"', response_text)
    if partial:
        if len(partial) < expected_len:
            partial += ["O"] * (expected_len - len(partial))
        else:
            partial = partial[:expected_len]
        return partial

    print(f"  [WARNING] Could not parse labels. Defaulting to all-O.")
    return ["O"] * expected_len


# ------------------------------------------------------------------ #
# Main inference loop
# ------------------------------------------------------------------ #

def run_inference(cfg: dict, split: str, strategy: str, model_id_override: str | None = None) -> None:
    model_cfg  = cfg["model"]
    data_cfg   = cfg["data"]
    prompt_cfg = cfg["prompting"]
    out_cfg    = cfg["output"]
    label_cfg  = cfg["labels"]

    valid_labels = set(label_cfg["valid_set"])
    model_id  = model_id_override or model_cfg.get("hf_model_id", model_cfg["name"])
    hf_cache  = model_cfg.get("hf_cache", "hf_cache")
    model_tag = model_id.replace(":", "-").replace("/", "-")

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
            train_sents,
            prompt_cfg["num_few_shot_examples"],
            contrastive=prompt_cfg.get("contrastive", False),
        )
        print(f"  Selected {len(few_shot_examples)} examples.")
        for ex in few_shot_examples:
            print(f"    [{ex.id}] spans={ex.error_spans()} tokens={ex.tokens[:8]}")

    # Prepare output path
    pred_path = out_cfg["predictions_file"].format(
        model_tag=model_tag, split=split, strategy=strategy
    )
    Path(pred_path).parent.mkdir(parents=True, exist_ok=True)

    # Resume from partial if available
    partial_path = pred_path + ".partial"
    results: list[dict] = []
    completed_ids: set[int] = set()

    if Path(partial_path).exists():
        print(f"Found partial predictions: {partial_path}")
        with open(partial_path, encoding="utf-8") as fh:
            partial_data = json.load(fh)
        results = partial_data["predictions"]
        completed_ids = {r["sentence_id"] for r in results}
        print(f"  Resuming from {len(completed_ids)} completed sentences.")

    sentences = [s for s in sentences if s.id not in completed_ids]
    n = len(sentences)
    print(f"  {n} sentences remaining.")

    if n == 0:
        print("All sentences already processed.")
    else:
        # Load model
        t0 = time.time()
        tokenizer, model = load_model(model_id, hf_cache)
        print(f"  Model loaded in {time.time() - t0:.1f}s")

        batch_size = estimate_batch_size(model)

        # Determine max_new_tokens based on average sentence length
        avg_len = sum(len(s.tokens) for s in sentences) / len(sentences)
        max_new_tokens = min(int(avg_len * 6) + 50, 1024)
        print(f"  max_new_tokens: {max_new_tokens} (avg sentence len: {avg_len:.1f})")

        # Inference loop
        t1 = time.time()
        for batch_start in range(0, n, batch_size):
            batch = sentences[batch_start: batch_start + batch_size]

            if batch_start % 500 == 0 or batch_start == 0:
                elapsed = time.time() - t1
                print(f"  [{batch_start}/{n}] elapsed: {elapsed:.0f}s ...")

            # Build prompts
            prompts = []
            for sent in batch:
                if strategy == "few_shot":
                    user = build_few_shot_prompt(sent, few_shot_examples)
                else:
                    user = build_zero_shot_prompt(sent)
                prompts.append(format_chat_prompt(user, tokenizer, model_id))

            # Generate
            responses = generate_batch(prompts, tokenizer, model, max_new_tokens)

            # Parse and store
            for sent, response in zip(batch, responses):
                pred_labels = parse_labels(response, len(sent.tokens))
                results.append({
                    "sentence_id": sent.id,
                    "tokens":      sent.tokens,
                    "gold_labels": sent.labels,
                    "pred_labels": pred_labels,
                    "raw_response": response,
                })

            # Checkpoint every 100 sentences
            if len(results) % 100 < batch_size:
                with open(partial_path, "w", encoding="utf-8") as fh:
                    json.dump({
                        "model": model_id,
                        "split": split,
                        "strategy": strategy,
                        "num_sentences": len(results),
                        "predictions": results,
                    }, fh, ensure_ascii=False, indent=2)

        elapsed = time.time() - t1
        print(f"\nInference complete in {elapsed:.1f}s "
              f"({elapsed / n:.2f}s/sentence)")

    # Save final predictions
    with open(partial_path, "w", encoding="utf-8") as fh:
        json.dump({
            "model": model_id,
            "split": split,
            "strategy": strategy,
            "num_sentences": len(results),
            "predictions": results,
        }, fh, ensure_ascii=False, indent=2)
    os.rename(partial_path, pred_path)
    print(f"Predictions saved to: {pred_path}")


# ------------------------------------------------------------------ #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GED-PT SLM inference via HuggingFace Transformers"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument(
        "--strategy", default=None,
        help="zero_shot or few_shot (overrides config)"
    )
    parser.add_argument(
        "--model", default=None,
        help="HuggingFace model ID (overrides config)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    strategy = args.strategy or cfg["prompting"]["strategy"]
    run_inference(cfg, args.split, strategy, model_id_override=args.model)