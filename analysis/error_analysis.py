"""
error_analysis.py
Per-taxonomy-category analysis of model predictions for Brazilian Portuguese GED.

For Ollama-based models (zero_shot, few_shot):
  Computes span-level Precision, Recall, F1 per taxonomy category.

For two-stage Stage 1 (binary detection):
  Computes sentence-level detection rate per taxonomy category —
  i.e., what fraction of sentences containing each error type did
  Stage 1 correctly flag as containing an error?

Usage:
    python error_analysis.py
    python error_analysis.py --taxonomy data/test_bio_taxonomy.tsv --output analysis/
"""

from __future__ import annotations
import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import re


# ------------------------------------------------------------------ #
# Data loading
# ------------------------------------------------------------------ #

_SENTENCE_RE = re.compile(r"^Sentença", re.IGNORECASE)


@dataclass
class SentenceInfo:
    id: int
    tokens: list[str]
    labels: list[str]
    taxonomy: list[str]   # fourth column from taxonomy file

    def error_spans(self) -> list[tuple[int, int]]:
        spans = []
        start = None
        for i, lbl in enumerate(self.labels):
            if lbl == "B-WRONG":
                if start is not None:
                    spans.append((start, i - 1))
                start = i
            elif lbl == "O" and start is not None:
                spans.append((start, i - 1))
                start = None
        if start is not None:
            spans.append((start, len(self.labels) - 1))
        return spans

    def span_category(self, start: int, end: int) -> str:
        """Return the taxonomy category for a span (use B-WRONG token's category)."""
        return self.taxonomy[start] if start < len(self.taxonomy) else "UNKNOWN"

    def has_error(self) -> bool:
        return any(l != "O" for l in self.labels)

    def error_categories(self) -> set[str]:
        """Return set of taxonomy categories present in this sentence."""
        cats = set()
        for start, end in self.error_spans():
            cats.add(self.span_category(start, end))
        return cats


def load_taxonomy_file(path: str | Path) -> dict[int, SentenceInfo]:
    """Load 4-column taxonomy TSV. Returns dict keyed by sentence_id (1-based)."""
    sentences: dict[int, SentenceInfo] = {}
    sentence_id = 0
    tokens, labels, taxonomy = [], [], []

    def flush():
        if tokens:
            sentences[sentence_id] = SentenceInfo(
                id=sentence_id,
                tokens=list(tokens),
                labels=list(labels),
                taxonomy=list(taxonomy),
            )

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if _SENTENCE_RE.match(line):
                flush()
                sentence_id += 1
                tokens.clear(); labels.clear(); taxonomy.clear()
            elif not line.strip():
                flush()
                tokens.clear(); labels.clear(); taxonomy.clear()
            else:
                parts = line.split("\t")
                if len(parts) >= 4:
                    tokens.append(parts[0])
                    labels.append(parts[1].strip())
                    taxonomy.append(parts[3].strip())
                elif len(parts) >= 2:
                    tokens.append(parts[0])
                    labels.append(parts[1].strip())
                    taxonomy.append("-")
    flush()
    return sentences


def load_predictions(path: str | Path) -> list[dict]:
    """Load a predictions JSON file and return the predictions list."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["predictions"]


def load_stage1_cache(path: str | Path) -> dict[int, bool]:
    """Load a Stage 1 cache JSON. Returns dict of sentence_id -> has_error."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}


# ------------------------------------------------------------------ #
# Span-level per-category metrics
# ------------------------------------------------------------------ #

@dataclass
class CategoryMetrics:
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


def extract_pred_spans(pred_labels: list[str]) -> set[tuple[int, int]]:
    """Extract predicted error spans as (start, end_inclusive) tuples."""
    spans = set()
    start = None
    for i, lbl in enumerate(pred_labels):
        if lbl == "B-WRONG":
            if start is not None:
                spans.add((start, i - 1))
            start = i
        elif lbl != "I-WRONG" and start is not None:
            spans.add((start, i - 1))
            start = None
    if start is not None:
        spans.add((start, len(pred_labels) - 1))
    return spans


def compute_per_category_metrics(
    predictions: list[dict],
    taxonomy_sents: dict[int, SentenceInfo],
) -> dict[str, CategoryMetrics]:
    """
    Compute span-level TP/FP/FN per taxonomy category.

    A predicted span is a TP for category C if it exactly matches a gold span
    whose taxonomy category is C.
    A predicted span with no gold match is a FP — assigned to category
    'NO_MATCH' (the model invented an error that doesn't exist).
    A gold span with no predicted match is a FN for its taxonomy category.
    """
    metrics: dict[str, CategoryMetrics] = defaultdict(CategoryMetrics)

    for pred in predictions:
        sid = pred["sentence_id"]
        pred_labels = pred["pred_labels"]
        sent = taxonomy_sents.get(sid)
        if sent is None:
            continue

        gold_spans = {(s, e): sent.span_category(s, e)
                      for s, e in sent.error_spans()}
        pred_spans = extract_pred_spans(pred_labels)

        matched_gold: set[tuple[int, int]] = set()

        # Check each predicted span
        for ps in pred_spans:
            if ps in gold_spans:
                # TP: correct span, correct category
                cat = gold_spans[ps]
                metrics[cat].tp += 1
                matched_gold.add(ps)
            else:
                # FP: predicted span has no gold match
                metrics["NO_MATCH_pred"].fp += 1

        # Unmatched gold spans are FNs
        for gs, cat in gold_spans.items():
            if gs not in matched_gold:
                metrics[cat].fn += 1

    return dict(metrics)


# ------------------------------------------------------------------ #
# Stage 1 per-category detection rate
# ------------------------------------------------------------------ #

@dataclass
class DetectionStats:
    sentences_with_category: int = 0
    sentences_detected: int = 0
    sentences_missed: int = 0
    false_positives: int = 0   # sentences flagged but have no error of this category

    @property
    def detection_rate(self) -> float:
        total = self.sentences_with_category
        return self.sentences_detected / total if total > 0 else 0.0

    def as_dict(self) -> dict:
        return {
            "detection_rate": round(self.detection_rate, 4),
            "sentences_with_category": self.sentences_with_category,
            "sentences_detected": self.sentences_detected,
            "sentences_missed": self.sentences_missed,
        }


def compute_stage1_detection(
    stage1_cache: dict[int, bool],
    taxonomy_sents: dict[int, SentenceInfo],
) -> dict[str, DetectionStats]:
    """
    For each taxonomy category, compute the fraction of sentences
    containing that error type that Stage 1 correctly flagged.
    """
    stats: dict[str, DetectionStats] = defaultdict(DetectionStats)

    for sid, sent in taxonomy_sents.items():
        if not sent.has_error():
            continue

        pred_has_error = stage1_cache.get(sid, True)
        cats = sent.error_categories()

        for cat in cats:
            stats[cat].sentences_with_category += 1
            if pred_has_error:
                stats[cat].sentences_detected += 1
            else:
                stats[cat].sentences_missed += 1

    return dict(stats)


# ------------------------------------------------------------------ #
# Reporting
# ------------------------------------------------------------------ #

def print_span_analysis(
    model_name: str,
    metrics: dict[str, CategoryMetrics],
) -> None:
    print(f"\n{'='*70}")
    print(f"  Model: {model_name}")
    print(f"{'='*70}")

    # Sort by F1 descending
    sorted_cats = sorted(
        [(cat, m) for cat, m in metrics.items()],
        key=lambda x: -x[1].f1
    )

    print(f"  {'Category':<35} {'P':>7} {'R':>7} {'F1':>7} {'TP':>6} {'FP':>6} {'FN':>6}")
    print(f"  {'-'*70}")
    for cat, m in sorted_cats:
        print(f"  {cat:<35} {m.precision:>7.4f} {m.recall:>7.4f} {m.f1:>7.4f} "
              f"{m.tp:>6} {m.fp:>6} {m.fn:>6}")

    # Overall
    total_tp = sum(m.tp for m in metrics.values())
    total_fp = sum(m.fp for m in metrics.values())
    total_fn = sum(m.fn for m in metrics.values())
    overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0
    print(f"  {'-'*70}")
    print(f"  {'OVERALL':<35} {overall_p:>7.4f} {overall_r:>7.4f} {overall_f1:>7.4f} "
          f"{total_tp:>6} {total_fp:>6} {total_fn:>6}")


def print_stage1_analysis(
    model_name: str,
    stats: dict[str, DetectionStats],
) -> None:
    print(f"\n{'='*70}")
    print(f"  Stage 1 Detection: {model_name}")
    print(f"{'='*70}")
    print(f"  {'Category':<35} {'DetRate':>9} {'Detected':>9} {'Missed':>9} {'Total':>9}")
    print(f"  {'-'*60}")

    sorted_cats = sorted(stats.items(), key=lambda x: -x[1].detection_rate)
    for cat, s in sorted_cats:
        print(f"  {cat:<35} {s.detection_rate:>9.4f} {s.sentences_detected:>9} "
              f"{s.sentences_missed:>9} {s.sentences_with_category:>9}")


def save_results(
    output_dir: Path,
    model_name: str,
    span_metrics: dict[str, CategoryMetrics] | None = None,
    stage1_stats: dict[str, DetectionStats] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.replace("/", "-").replace(":", "-")

    if span_metrics is not None:
        out = {cat: m.as_dict() for cat, m in span_metrics.items()}
        path = output_dir / f"{safe_name}_span_metrics.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    if stage1_stats is not None:
        out = {cat: s.as_dict() for cat, s in stage1_stats.items()}
        path = output_dir / f"{safe_name}_stage1_detection.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)


# ------------------------------------------------------------------ #
# Cross-model comparison table
# ------------------------------------------------------------------ #

def print_comparison_table(
    all_metrics: dict[str, dict[str, CategoryMetrics]],
    metric: str = "f1",
) -> None:
    """Print a cross-model comparison table for a given metric (f1/precision/recall)."""
    all_cats = sorted(set(
        cat for m in all_metrics.values()
        for cat in m.keys()
        if cat != "NO_MATCH_pred"
    ))

    model_names = list(all_metrics.keys())
    col_width = 10

    print(f"\n{'='*80}")
    print(f"  Cross-model comparison — {metric.upper()} per category")
    print(f"{'='*80}")

    header = f"  {'Category':<35}" + "".join(f"{n[:col_width-1]:>{col_width}}" for n in model_names)
    print(header)
    print(f"  {'-'*80}")

    for cat in all_cats:
        row = f"  {cat:<35}"
        for model, metrics in all_metrics.items():
            if cat in metrics:
                val = getattr(metrics[cat], metric)
                row += f"{val:>{col_width}.4f}"
            else:
                row += f"{'N/A':>{col_width}}"
        print(row)


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-category error analysis")
    parser.add_argument("--taxonomy", default="data/test_bio_taxonomy.tsv")
    parser.add_argument("--predictions_dir", default="predictions")
    parser.add_argument("--output", default="analysis")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load taxonomy
    taxonomy_path = Path(args.taxonomy)
    if not taxonomy_path.exists():
        print(f"ERROR: taxonomy file not found: {taxonomy_path}")
        print("Run cogroo_taxonomy.py first to generate it.")
        exit(1)

    print(f"Loading taxonomy from {taxonomy_path}...")
    taxonomy_sents = load_taxonomy_file(taxonomy_path)
    print(f"  {len(taxonomy_sents)} sentences loaded.")

    pred_dir = Path(args.predictions_dir)
    output_dir = Path(args.output)

    # ---- Ollama-based prediction files ----
    # Skip partial files and qwen3-14b partial
    prediction_files = [
        ("aya-expanse:8b zero_shot",      "aya-expanse-8b_test_zero_shot.json"),
        ("aya-expanse:8b few_shot",        "aya-expanse-8b_test_few_shot.json"),
        ("deepseek-r1:7b zero_shot",       "deepseek-r1-7b_test_zero_shot.json"),
        ("gemma3:12b zero_shot",           "gemma3-12b_test_zero_shot.json"),
        ("gemma3:12b few_shot",            "gemma3-12b_test_few_shot.json"),
        ("gemma4:e2b zero_shot",           "gemma4-e2b_test_zero_shot.json"),
        ("gemma4:e2b few_shot",            "gemma4-e2b_test_few_shot.json"),
        ("qwen3:8b zero_shot",             "qwen3-8b_test_zero_shot.json"),
        ("qwen3:8b few_shot",              "qwen3-8b_test_few_shot.json"),
        ("Qwen3-8B two_stage",             "Qwen-Qwen3-8B_test_two_stage.json"),
    ]

    all_span_metrics: dict[str, dict[str, CategoryMetrics]] = {}

    for model_name, filename in prediction_files:
        path = pred_dir / filename
        if not path.exists():
            print(f"  Skipping {model_name} — file not found: {path}")
            continue

        print(f"\nAnalyzing: {model_name}")
        predictions = load_predictions(path)
        metrics = compute_per_category_metrics(predictions, taxonomy_sents)
        all_span_metrics[model_name] = metrics

        print_span_analysis(model_name, metrics)
        save_results(output_dir, model_name, span_metrics=metrics)

    # ---- Stage 1 cache files ----
    stage1_files = [
        ("Qwen3-8B Stage1",      "Qwen-Qwen3-8B_test_stage1_cache.json"),
        ("gemma3-12b-it Stage1", "google-gemma-3-12b-it_test_stage1_cache.json"),
    ]

    for model_name, filename in stage1_files:
        path = pred_dir / filename
        if not path.exists():
            print(f"  Skipping {model_name} — file not found: {path}")
            continue

        print(f"\nStage 1 analysis: {model_name}")
        stage1_cache = load_stage1_cache(path)
        stage1_stats = compute_stage1_detection(stage1_cache, taxonomy_sents)

        print_stage1_analysis(model_name, stage1_stats)
        save_results(output_dir, model_name, stage1_stats=stage1_stats)

    # ---- Cross-model comparison ----
    if len(all_span_metrics) > 1:
        print_comparison_table(all_span_metrics, metric="f1")
        print_comparison_table(all_span_metrics, metric="recall")
        print_comparison_table(all_span_metrics, metric="precision")

    print(f"\nResults saved to: {output_dir}/")