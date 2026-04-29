"""
evaluate.py
Span-level Precision, Recall, and F1 for Grammar Error Detection (BIO scheme).

A predicted error span must exactly match a gold span (same start and end token
indices within the same sentence) to count as a true positive.
This is the standard "exact span match" metric used in GED/GEC evaluation.

Usage:
    python evaluate.py --predictions predictions/gemma3-12b_test_zero_shot.json
    python evaluate.py --predictions predictions/qwen2.5-14b_test_few_shot.json --verbose
"""

from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from pathlib import Path


# ------------------------------------------------------------------ #
# Span extraction
# ------------------------------------------------------------------ #

@dataclass(frozen=True)
class ErrorSpan:
    sentence_id: int
    start: int   # token index (0-based within sentence)
    end: int     # token index, inclusive

    def __repr__(self) -> str:
        return f"Span(sent={self.sentence_id}, [{self.start}:{self.end}])"


def extract_spans(sentence_id: int, labels: list[str]) -> set[ErrorSpan]:
    """Convert a BIO label sequence into a set of ErrorSpan objects."""
    spans: set[ErrorSpan] = set()
    start: int | None = None

    for i, lbl in enumerate(labels):
        if lbl == "B-WRONG":
            if start is not None:
                spans.add(ErrorSpan(sentence_id, start, i - 1))
            start = i
        elif lbl == "I-WRONG":
            if start is None:
                # Orphan I-WRONG: treat as beginning of span (lenient)
                start = i
        else:  # O or unexpected
            if start is not None:
                spans.add(ErrorSpan(sentence_id, start, i - 1))
                start = None

    if start is not None:
        spans.append if False else spans.add(ErrorSpan(sentence_id, start, len(labels) - 1))

    return spans


# ------------------------------------------------------------------ #
# Metrics
# ------------------------------------------------------------------ #

@dataclass
class SpanMetrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def as_dict(self) -> dict:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
        }


def compute_span_metrics(predictions: list[dict]) -> SpanMetrics:
    """Compute aggregate span-level metrics over all sentences."""
    metrics = SpanMetrics()
    for item in predictions:
        sid = item["sentence_id"]
        gold_spans = extract_spans(sid, item["gold_labels"])
        pred_spans = extract_spans(sid, item["pred_labels"])

        metrics.tp += len(gold_spans & pred_spans)
        metrics.fp += len(pred_spans - gold_spans)
        metrics.fn += len(gold_spans - pred_spans)

    return metrics


# ------------------------------------------------------------------ #
# Per-sentence breakdown (verbose)
# ------------------------------------------------------------------ #

def sentence_breakdown(predictions: list[dict], top_n: int = 20) -> None:
    """Print per-sentence TP/FP/FN for the worst-performing sentences."""
    rows = []
    for item in predictions:
        sid = item["sentence_id"]
        gold = extract_spans(sid, item["gold_labels"])
        pred = extract_spans(sid, item["pred_labels"])
        tp = len(gold & pred)
        fp = len(pred - gold)
        fn = len(gold - pred)
        if fp + fn > 0:
            rows.append((fp + fn, sid, tp, fp, fn, item["tokens"]))

    rows.sort(reverse=True)
    print(f"\nTop {min(top_n, len(rows))} sentences by errors (FP+FN):")
    print(f"{'SentID':>8}  {'TP':>4}  {'FP':>4}  {'FN':>4}  Text")
    print("-" * 72)
    for _, sid, tp, fp, fn, tokens in rows[:top_n]:
        text = " ".join(tokens)[:55]
        print(f"{sid:>8}  {tp:>4}  {fp:>4}  {fn:>4}  {text}")


# ------------------------------------------------------------------ #
# Label-level token accuracy (supplementary)
# ------------------------------------------------------------------ #

def token_accuracy(predictions: list[dict]) -> float:
    correct = total = 0
    for item in predictions:
        for g, p in zip(item["gold_labels"], item["pred_labels"]):
            correct += int(g == p)
            total += 1
    return correct / total if total else 0.0


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GED-PT span-level evaluation")
    parser.add_argument(
        "--predictions", required=True, help="Path to JSON predictions file"
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional path to save metrics JSON (auto-derived from predictions path if omitted)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-sentence breakdown for error analysis"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pred_path = Path(args.predictions)

    with open(pred_path, encoding="utf-8") as fh:
        data = json.load(fh)

    preds = data["predictions"]
    model = data.get("model", "unknown")
    split = data.get("split", "unknown")
    strategy = data.get("strategy", "unknown")

    # Span-level metrics
    span_m = compute_span_metrics(preds)

    # Token accuracy (supplementary)
    tok_acc = token_accuracy(preds)

    result = {
        "model": model,
        "split": split,
        "strategy": strategy,
        "num_sentences": len(preds),
        "span_metrics": span_m.as_dict(),
        "token_accuracy": round(tok_acc, 4),
    }

    # Print summary
    print("\n" + "=" * 50)
    print(f"  Model    : {model}")
    print(f"  Split    : {split}")
    print(f"  Strategy : {strategy}")
    print(f"  Sentences: {len(preds)}")
    print("=" * 50)
    print(f"  Span Precision : {span_m.precision:.4f}")
    print(f"  Span Recall    : {span_m.recall:.4f}")
    print(f"  Span F1        : {span_m.f1:.4f}")
    print(f"  (TP={span_m.tp}, FP={span_m.fp}, FN={span_m.fn})")
    print(f"  Token Accuracy : {tok_acc:.4f}")
    print("=" * 50)

    # Verbose breakdown
    if args.verbose:
        sentence_breakdown(preds)

    # Save metrics
    out_path = args.output
    if out_path is None:
        out_path = str(pred_path).replace("predictions/", "metrics/").replace(".json", "_metrics.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(f"\nMetrics saved to: {out_path}")
