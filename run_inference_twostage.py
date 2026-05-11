"""
run_inference_2stage.py
Two-stage Grammar Error Detection for Brazilian Portuguese using HuggingFace Transformers.

Stage 1 — Binary detection: does this sentence contain a grammatical error?
           Output: {"has_error": true/false}
Stage 2 — Span localization (only if has_error=true): which tokens are wrong?
           Output: {"labels": ["O", "B-WRONG", ...]}

This approach reduces cognitive load on the model by separating detection from localization.
Uses HuggingFace Transformers with dynamic batching for significantly faster inference
than Ollama's sequential HTTP calls.

Supported models (HuggingFace IDs):
    Qwen/Qwen3-8B
    google/gemma-3-12b-it
    TucanoBR/Tucano-2b4-Instruct
    deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

Usage:
    python run_inference_2stage.py --model Qwen/Qwen3-8B --split test
    python run_inference_2stage.py --model TucanoBR/Tucano-2b4-Instruct --split test
    python run_inference_2stage.py --model google/gemma-3-12b-it --split test --batch_size 8
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
# Prompt templates — Stage 1: binary detection
# ------------------------------------------------------------------ #

STAGE1_SYSTEM = """\
You are a Brazilian Portuguese grammar checker.
Your task is to decide whether a sentence contains a grammatical error.
Output ONLY a valid JSON object: {"has_error": true} or {"has_error": false}
Do NOT output any explanation or extra text."""

STAGE1_USER_TEMPLATE = """\
Sentence tokens: {tokens_json}

Does this sentence contain a grammatical error?
Respond with ONLY: {{"has_error": true}} or {{"has_error": false}}"""


# ------------------------------------------------------------------ #
# Prompt templates — Stage 2: span localization
# ------------------------------------------------------------------ #

STAGE2_SYSTEM = """\
You are a Brazilian Portuguese grammar error localizer.
A sentence has been flagged as containing a grammatical error.
Your task is to label each token with one of:
  - O        : grammatically correct token
  - B-WRONG  : FIRST token of an error span
  - I-WRONG  : continuation of an error span

Rules:
1. Every token must receive exactly one label.
2. Error spans always start with B-WRONG followed by zero or more I-WRONG.
3. Output ONLY a valid JSON object: {"labels": ["O", "B-WRONG", ...]}
4. The labels list must have the SAME length as the input tokens list.
Do NOT output any explanation or extra text."""

STAGE2_USER_TEMPLATE = """\
Tokens: {tokens_json}

Label each token. Respond with ONLY: {{"labels": [...]}}"""


# ------------------------------------------------------------------ #
# Model loading
# ------------------------------------------------------------------ #

def load_model(model_id: str, hf_cache: str) -> tuple:
    """Load tokenizer and model onto GPU with automatic device mapping."""
    print(f"Loading model: {model_id}")
    print(f"  HF_HOME: {hf_cache}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=hf_cache,
        trust_remote_code=True,
        padding_side="left",   # left padding for decoder-only batch inference
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=hf_cache,
        torch_dtype=torch.bfloat16,   # bf16 on A100 — fastest + numerically stable
        device_map="auto",             # automatic GPU/CPU placement
        trust_remote_code=True,
    )
    model.eval()
    print(f"  Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    return tokenizer, model


# ------------------------------------------------------------------ #
# Dynamic batch sizing
# ------------------------------------------------------------------ #

def estimate_batch_size(model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> int:
    """
    Estimate a safe batch size based on available GPU VRAM.
    Uses a conservative heuristic: 1GB VRAM per batch item for 8b models,
    scaled for larger models.
    """
    if not torch.cuda.is_available():
        return 1

    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
    model_params_b = sum(p.numel() for p in model.parameters()) / 1e9
    # rough estimate: model uses ~2 bytes per param in bf16
    model_vram = model_params_b * 2
    available = total_vram - model_vram - 2.0  # 2GB safety margin

    # each batch item needs ~0.5-1GB for activations at typical sentence length
    batch_size = max(1, int(available / 1.0))
    batch_size = min(batch_size, 32)  # cap at 32 regardless
    print(f"  Dynamic batch size: {batch_size} "
          f"(VRAM: {total_vram:.1f}GB, model: {model_vram:.1f}GB, available: {available:.1f}GB)")
    return batch_size


# ------------------------------------------------------------------ #
# Chat prompt formatting
# ------------------------------------------------------------------ #

def format_chat_prompt(system: str, user: str, tokenizer: AutoTokenizer, model_id: str) -> str:
    """
    Format a system+user prompt using the model's chat template if available,
    falling back to a generic format for models without one (e.g. Tucano base).
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        # Most instruct models have a chat template
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
    except Exception:
        # Fallback for models without chat template (e.g. Tucano base variants)
        prompt = f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"
    return prompt


# ------------------------------------------------------------------ #
# Batch generation
# ------------------------------------------------------------------ #

def generate_batch(
    prompts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> list[str]:
    """
    Run a batch of prompts through the model and return generated text per prompt.
    Uses greedy decoding (temperature=0) for deterministic output.
    """
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
            do_sample=False,          # greedy — deterministic
            temperature=None,         # must be None when do_sample=False
            top_p=None,               # must be None when do_sample=False
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (not the input prompt)
    responses = []
    for i, output in enumerate(outputs):
        input_len = inputs["input_ids"].shape[1]
        new_tokens = output[input_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        responses.append(text)

    return responses


# ------------------------------------------------------------------ #
# Response parsers
# ------------------------------------------------------------------ #

_HAS_ERROR_RE = re.compile(r'\{.*?"has_error"\s*:\s*(true|false).*?\}', re.DOTALL | re.IGNORECASE)
_LABELS_RE = re.compile(r'\{.*?"labels"\s*:\s*\[.*?\]\s*\}', re.DOTALL)

VALID_LABELS = {"O", "B-WRONG", "I-WRONG"}


def parse_has_error(response: str) -> bool:
    """Parse stage 1 response. Defaults to True on parse failure (conservative)."""
    match = _HAS_ERROR_RE.search(response)
    if match:
        try:
            obj = json.loads(match.group())
            return bool(obj.get("has_error", True))
        except json.JSONDecodeError:
            pass
    # fallback: look for true/false anywhere in response
    lower = response.lower()
    if "false" in lower:
        return False
    return True  # conservative default: assume error present


def parse_labels(response: str, expected_len: int) -> list[str]:
    """Parse stage 2 response. Falls back to all-O on failure."""
    match = _LABELS_RE.search(response)
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
    print(f"  [WARNING] Could not parse labels from response. Defaulting to all-O.")
    return ["O"] * expected_len


# ------------------------------------------------------------------ #
# Main two-stage inference loop
# ------------------------------------------------------------------ #

def run_2stage_inference(
    sentences: list[Sentence],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    model_id: str,
    batch_size: int,
    pred_path: str,
    max_new_tokens_s1: int = 32,
    max_new_tokens_s2: int = 512,
) -> list[dict]:

    partial_path = pred_path + ".partial"
    model_tag = model_id.replace("/", "-").replace(":", "-")
    split = Path(pred_path).stem.split("_")[1]  # extract split from filename

    stage1_cache_path = f"predictions/{model_tag}_{split}_stage1_cache.json"
    stage2_cache_path = f"predictions/{model_tag}_{split}_stage2_cache.json"

    results: list[dict] = []
    completed_ids: set[int] = set()

    # Resume from partial final results if available
    if Path(partial_path).exists():
        print(f"Found partial predictions: {partial_path}")
        with open(partial_path, encoding="utf-8") as fh:
            partial_data = json.load(fh)
        results = partial_data["predictions"]
        completed_ids = {r["sentence_id"] for r in results}
        print(f"  Resuming from {len(completed_ids)} completed sentences.")

    remaining = [s for s in sentences if s.id not in completed_ids]
    n = len(remaining)
    print(f"  {n} sentences to process.")

    # ---- Stage 1: load cache or run ----
    stage1_results: dict[int, bool] = {}

    if Path(stage1_cache_path).exists():
        print(f"Found Stage 1 cache: {stage1_cache_path}")
        with open(stage1_cache_path, encoding="utf-8") as fh:
            stage1_results = {int(k): v for k, v in json.load(fh).items()}
        # only keep results for sentences not yet in final results
        stage1_results = {
            sid: v for sid, v in stage1_results.items()
            if sid not in completed_ids
        }
        print(f"  Loaded {len(stage1_results)} Stage 1 results from cache.")
        # sentences still needing Stage 1
        needs_stage1 = [s for s in remaining if s.id not in stage1_results]
    else:
        needs_stage1 = remaining

    if needs_stage1:
        print(f"\n[Stage 1] Binary detection for {len(needs_stage1)} sentences "
              f"(batch_size={batch_size}) ...")
        for batch_start in range(0, len(needs_stage1), batch_size):
            batch = needs_stage1[batch_start: batch_start + batch_size]
            if batch_start % 500 == 0 or batch_start == 0:
                print(f"  Stage 1: [{batch_start}/{len(needs_stage1)}] ...")

            prompts = [
                format_chat_prompt(
                    STAGE1_SYSTEM,
                    STAGE1_USER_TEMPLATE.format(
                        tokens_json=json.dumps(s.tokens, ensure_ascii=False)
                    ),
                    tokenizer,
                    model_id,
                )
                for s in batch
            ]
            responses = generate_batch(
                prompts, tokenizer, model, max_new_tokens=max_new_tokens_s1
            )
            for sent, response in zip(batch, responses):
                stage1_results[sent.id] = parse_has_error(response)

            # Checkpoint Stage 1 cache every batch
            with open(stage1_cache_path, "w", encoding="utf-8") as fh:
                json.dump(stage1_results, fh)

    n_errors = sum(stage1_results.values())
    print(f"  Stage 1 complete: {n_errors} errors, "
          f"{len(stage1_results) - n_errors} correct.")

    # ---- Stage 2: load cache or run ----
    stage2_results: dict[int, list[str]] = {}

    if Path(stage2_cache_path).exists():
        print(f"Found Stage 2 cache: {stage2_cache_path}")
        with open(stage2_cache_path, encoding="utf-8") as fh:
            stage2_results = {int(k): v for k, v in json.load(fh).items()}
        stage2_results = {
            sid: v for sid, v in stage2_results.items()
            if sid not in completed_ids
        }
        print(f"  Loaded {len(stage2_results)} Stage 2 results from cache.")

    flagged = [
        s for s in remaining
        if stage1_results.get(s.id, True) and s.id not in stage2_results
    ]

    if flagged:
        print(f"\n[Stage 2] Span localization for {len(flagged)} sentences "
              f"(batch_size={batch_size}) ...")
        for batch_start in range(0, len(flagged), batch_size):
            batch = flagged[batch_start: batch_start + batch_size]
            if batch_start % 500 == 0 or batch_start == 0:
                print(f"  Stage 2: [{batch_start}/{len(flagged)}] ...")

            prompts = [
                format_chat_prompt(
                    STAGE2_SYSTEM,
                    STAGE2_USER_TEMPLATE.format(
                        tokens_json=json.dumps(s.tokens, ensure_ascii=False)
                    ),
                    tokenizer,
                    model_id,
                )
                for s in batch
            ]
            responses = generate_batch(
                prompts, tokenizer, model, max_new_tokens=max_new_tokens_s2
            )
            for sent, response in zip(batch, responses):
                stage2_results[sent.id] = parse_labels(response, len(sent.tokens))

            # Checkpoint Stage 2 cache every batch
            with open(stage2_cache_path, "w", encoding="utf-8") as fh:
                json.dump(stage2_results, fh)

    # ---- Assemble final results ----
    print(f"\nAssembling results ...")
    for sent in remaining:
        has_error = stage1_results.get(sent.id, True)
        pred_labels = (
            stage2_results.get(sent.id, ["O"] * len(sent.tokens))
            if has_error
            else ["O"] * len(sent.tokens)
        )
        results.append({
            "sentence_id": sent.id,
            "tokens": sent.tokens,
            "gold_labels": sent.labels,
            "pred_labels": pred_labels,
            "stage1_has_error": has_error,
        })

        if len(results) % 100 == 0:
            _save_partial(results, partial_path, model_id)

    # Clean up stage caches on successful completion
    for cache in [stage1_cache_path, stage2_cache_path]:
        if Path(cache).exists():
            os.remove(cache)
            print(f"  Removed cache: {cache}")

    return results


def _save_partial(results: list[dict], partial_path: str, model_id: str) -> None:
    with open(partial_path, "w", encoding="utf-8") as fh:
        json.dump({
            "model": model_id,
            "strategy": "two_stage",
            "num_sentences": len(results),
            "predictions": results,
        }, fh, ensure_ascii=False, indent=2)


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GED-PT two-stage inference via HuggingFace")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-8B",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--split", default="test", choices=["test", "val"],
        help="Dataset split",
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config.yaml (used for data paths and output dirs)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=0,
        help="Batch size (0 = auto-detect from VRAM)",
    )
    parser.add_argument(
        "--hf_cache", default=None,
        help="HuggingFace cache directory (overrides HF_HOME env var)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    # Two-stage config section with CLI overrides
    ts_cfg = cfg.get("two_stage", {})
    model_id = args.model or ts_cfg.get("model", "Qwen/Qwen3-8B")
    hf_cache = args.hf_cache or ts_cfg.get("hf_cache", "hf_cache")
    batch_size = args.batch_size if args.batch_size > 0 else ts_cfg.get("batch_size", 0)
    max_new_tokens_s1 = ts_cfg.get("max_new_tokens_stage1", 32)
    max_new_tokens_s2 = ts_cfg.get("max_new_tokens_stage2", 512)

    data_cfg = cfg["data"]
    out_cfg = cfg["output"]
    label_cfg = cfg["labels"]

    os.makedirs(hf_cache, exist_ok=True)

    # Load data
    test_path = data_cfg[f"{args.split}_file"]
    print(f"Loading {args.split} split from: {test_path}")
    sentences = read_bio_file(test_path, valid_labels=set(label_cfg["valid_set"]))
    print(f"  {len(sentences)} sentences loaded.")

    # Prepare output path
    model_tag = model_id.replace("/", "-").replace(":", "-")
    pred_path = out_cfg["predictions_file"].format(
        model_tag=model_tag,
        split=args.split,
        strategy="two_stage",
    )
    Path(pred_path).parent.mkdir(parents=True, exist_ok=True)

    # Load model
    t0 = time.time()
    tokenizer, model = load_model(model_id, hf_cache)
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # Determine batch size
    if batch_size == 0:
        batch_size = estimate_batch_size(model, tokenizer)

    # Run two-stage inference
    t1 = time.time()
    results = run_2stage_inference(
        sentences, tokenizer, model, model_id, batch_size, pred_path,
        max_new_tokens_s1=max_new_tokens_s1,
        max_new_tokens_s2=max_new_tokens_s2,
    )
    elapsed = time.time() - t1
    print(f"\nInference complete in {elapsed:.1f}s "
          f"({elapsed / len(sentences):.2f}s/sentence)")

    # Save final predictions
    partial_path = pred_path + ".partial"
    with open(partial_path, "w", encoding="utf-8") as fh:
        json.dump({
            "model": model_id,
            "split": args.split,
            "strategy": "two_stage",
            "num_sentences": len(results),
            "predictions": results,
        }, fh, ensure_ascii=False, indent=2)
    os.rename(partial_path, pred_path)
    print(f"Predictions saved to: {pred_path}")

    # Run evaluation inline
    from evaluate import compute_span_metrics, token_accuracy

    metrics = compute_span_metrics(results)
    tok_acc = token_accuracy(results)

    print("\n" + "=" * 50)
    print(f"  Model    : {model_id}")
    print(f"  Split    : {args.split}")
    print(f"  Strategy : two_stage")
    print(f"  Sentences: {len(results)}")
    print("=" * 50)
    print(f"  Span Precision : {metrics.precision:.4f}")
    print(f"  Span Recall    : {metrics.recall:.4f}")
    print(f"  Span F1        : {metrics.f1:.4f}")
    print(f"  (TP={metrics.tp}, FP={metrics.fp}, FN={metrics.fn})")
    print(f"  Token Accuracy : {tok_acc:.4f}")

    # Stage 1 stats
    n_flagged = sum(1 for r in results if r["stage1_has_error"])
    print(f"\n  Stage 1 flagged as errors: {n_flagged}/{len(results)} "
          f"({100*n_flagged/len(results):.1f}%)")
    print("=" * 50)

    # Save metrics
    metrics_path = out_cfg["metrics_file"].format(
        model_tag=model_tag,
        split=args.split,
        strategy="two_stage",
    )
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump({
            "model": model_id,
            "split": args.split,
            "strategy": "two_stage",
            "num_sentences": len(results),
            "span_metrics": metrics.as_dict(),
            "token_accuracy": round(tok_acc, 4),
            "stage1_flagged": n_flagged,
            "stage1_flagged_pct": round(100 * n_flagged / len(results), 2),
        }, fh, ensure_ascii=False, indent=2)
    print(f"Metrics saved to: {metrics_path}")