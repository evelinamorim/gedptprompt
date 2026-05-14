"""
sample_false_negatives.py

Samples 300 false negative spans from a GED prediction JSON,
stratified by error category (from the unified taxonomy TSV),
and exports to CSV for manual error analysis.

Usage:
    python sample_false_negatives.py \
        --predictions predictions/Qwen-Qwen3-8B_test_two_stage.json \
        --taxonomy data/taxonomy.tsv \
        --output fn_sample_300.csv \
        --n 300 \
        --seed 42

The taxonomy TSV is expected to have columns:
    sentence_id  span_start  span_end  category
If you don't have it yet, use --no-taxonomy to skip stratification
by category and just stratify by span size instead (fallback).
"""

import json
import csv
import argparse
import random
from collections import defaultdict
from math import floor


# ---------------------------------------------------------------------------
# BIO helpers
# ---------------------------------------------------------------------------

def bio_to_spans(labels):
    """Convert a BIO label list to a list of (start, end) tuples (inclusive)."""
    spans = []
    start = None
    for i, label in enumerate(labels):
        if label == "B-WRONG":
            if start is not None:
                spans.append((start, i - 1))
            start = i
        elif label == "I-WRONG":
            if start is None:          # malformed: I without B — treat as B
                start = i
        else:                          # O
            if start is not None:
                spans.append((start, i - 1))
                start = None
    if start is not None:
        spans.append((start, len(labels) - 1))
    return spans


def spans_to_set(spans):
    return set(map(tuple, spans))


# ---------------------------------------------------------------------------
# False negative extraction
# ---------------------------------------------------------------------------

def extract_false_negatives(predictions):
    """
    For every sentence, find gold spans that are NOT in pred spans.
    Returns a list of dicts:
        sentence_id, span_start, span_end, span_tokens, gold_context
    """
    false_negatives = []
    for entry in predictions:
        sid        = entry["sentence_id"]
        tokens     = entry["tokens"]
        gold_spans = spans_to_set(bio_to_spans(entry["gold_labels"]))
        pred_spans = spans_to_set(bio_to_spans(entry["pred_labels"]))

        missed = gold_spans - pred_spans
        for (start, end) in missed:
            span_tokens = " ".join(tokens[start: end + 1])
            context     = " ".join(tokens)
            false_negatives.append({
                "sentence_id": sid,
                "span_start":  start,
                "span_end":    end,
                "span_tokens": span_tokens,
                "context":     context,
                "category":    "UNKNOWN",   # filled in later if taxonomy available
            })
    return false_negatives


# ---------------------------------------------------------------------------
# Taxonomy loading
# ---------------------------------------------------------------------------

def load_taxonomy(path):
    """
    Load a TSV with columns: sentence_id, span_start, span_end, category
    Returns a dict keyed by (sentence_id, span_start, span_end) -> category
    """
    taxonomy = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            key = (int(row["sentence_id"]), int(row["span_start"]), int(row["span_end"]))
            taxonomy[key] = row["category"]
    return taxonomy


def attach_categories(false_negatives, taxonomy):
    for fn in false_negatives:
        key = (fn["sentence_id"], fn["span_start"], fn["span_end"])
        fn["category"] = taxonomy.get(key, "UNKNOWN")
    return false_negatives


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def stratified_sample(false_negatives, n, seed=42):
    """
    Sample n items from false_negatives, proportionally stratified by category.
    Categories with fewer items than their quota are taken in full;
    the remaining quota is redistributed to larger categories.
    """
    rng = random.Random(seed)

    # Group by category
    by_category = defaultdict(list)
    for fn in false_negatives:
        by_category[fn["category"]].append(fn)

    total = len(false_negatives)
    if total == 0:
        raise ValueError("No false negatives found.")
    if n >= total:
        print(f"[warn] Requested {n} but only {total} FNs exist — returning all.")
        return false_negatives

    # Compute proportional quotas
    quotas = {
        cat: max(1, floor(n * len(items) / total))
        for cat, items in by_category.items()
    }

    # Adjust rounding so sum == n
    diff = n - sum(quotas.values())
    # Give extra slots to largest categories first
    for cat in sorted(by_category, key=lambda c: len(by_category[c]), reverse=True):
        if diff == 0:
            break
        quotas[cat] += 1
        diff -= 1

    # Sample
    sampled = []
    for cat, items in by_category.items():
        k = min(quotas[cat], len(items))
        sampled.extend(rng.sample(items, k))

    rng.shuffle(sampled)
    return sampled


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "sentence_id",
    "span_start",
    "span_end",
    "span_tokens",
    "category",
    "failure_mode",      # blank — annotator fills in
    "notes",             # blank — annotator fills in
    "context",
]

FAILURE_MODES = (
    "complete_miss | partial_span | boundary_shift | wrong_token"
)

def export_csv(sampled, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        # Write a comment row explaining failure_mode codes
        writer.writerow({
            "sentence_id": "# failure_mode codes:",
            "span_start":  FAILURE_MODES,
            "span_end":    "",
            "span_tokens": "",
            "category":    "",
            "failure_mode": "",
            "notes":        "",
            "context":      "",
        })
        for row in sampled:
            writer.writerow({k: row.get(k, "") for k in FIELDNAMES})
    print(f"Saved {len(sampled)} false negatives to {output_path}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(false_negatives, sampled):
    total_fns = len(false_negatives)
    print(f"\nTotal false negatives: {total_fns}")
    print(f"Sampled: {len(sampled)}\n")

    by_cat_total   = defaultdict(int)
    by_cat_sampled = defaultdict(int)
    for fn in false_negatives:
        by_cat_total[fn["category"]] += 1
    for fn in sampled:
        by_cat_sampled[fn["category"]] += 1

    print(f"{'Category':<30} {'Total FNs':>10} {'Sampled':>10} {'%Pop':>8}")
    print("-" * 62)
    for cat in sorted(by_cat_total, key=lambda c: -by_cat_total[c]):
        t = by_cat_total[cat]
        s = by_cat_sampled.get(cat, 0)
        pct = 100 * t / total_fns
        print(f"{cat:<30} {t:>10} {s:>10} {pct:>7.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sample false negatives for error analysis.")
    parser.add_argument("--predictions", required=True,
                        help="Path to prediction JSON (e.g. Qwen-Qwen3-8B_test_two_stage.json)")
    parser.add_argument("--taxonomy", default=None,
                        help="Path to taxonomy TSV (sentence_id, span_start, span_end, category). "
                             "Omit to label all spans as UNKNOWN.")
    parser.add_argument("--output", default="fn_sample_300.csv",
                        help="Output CSV path")
    parser.add_argument("--n", type=int, default=300,
                        help="Number of false negatives to sample (default: 300)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load predictions
    with open(args.predictions, encoding="utf-8") as f:
        data = json.load(f)
        print(data[0])
    predictions = data["predictions"]
    print(f"Loaded {len(predictions)} sentences from {args.predictions}")

    # Extract FNs
    false_negatives = extract_false_negatives(predictions)
    print(f"Found {len(false_negatives)} false negative spans")

    # Attach taxonomy categories if provided
    if args.taxonomy:
        taxonomy = load_taxonomy(args.taxonomy)
        false_negatives = attach_categories(false_negatives, taxonomy)
        print(f"Attached categories from {args.taxonomy}")
    else:
        print("No taxonomy provided — all spans labelled UNKNOWN. "
              "Stratification will be uniform.")

    # Stratified sample
    sampled = stratified_sample(false_negatives, args.n, seed=args.seed)

    # Summary
    print_summary(false_negatives, sampled)

    # Export
    export_csv(sampled, args.output)


if __name__ == "__main__":
    main()