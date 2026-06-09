"""
sample_taxonomy_validation.py
==============================
Draws a stratified sample of error spans from the taxonomy-annotated
train split for manual validation by a linguist.

For each sampled span the CSV contains:
  sentence_id       : sentence index (1-based)
  span_tokens       : the error tokens wrapped in **double asterisks**
  full_sentence     : full sentence text with the span highlighted
  taxonomy_auto     : automatically assigned taxonomy category
  is_error          : (empty — linguist fills in: yes / no / unsure)
  taxonomy_correct  : (empty — linguist fills in the correct category,
                       or leaves blank if taxonomy_auto is correct)
  notes             : (empty — free text)

Sampling strategy:
  - Minimum MIN_PER_CATEGORY spans per category (catches rare classes)
  - Remaining slots filled proportionally by category frequency
  - Total capped at TOTAL_SAMPLE
  - Random seed fixed for reproducibility

Usage:
    python sample_taxonomy_validation.py \\
        --input  data/train_bio_taxonomy.tsv \\
        --output data/taxonomy_validation_sample.csv
"""

from __future__ import annotations
import argparse
import csv
import random
import re
from collections import defaultdict
from pathlib import Path

# ------------------------------------------------------------------ #
# Parameters
# ------------------------------------------------------------------ #

TOTAL_SAMPLE      = 200
MIN_PER_CATEGORY  = 10
RANDOM_SEED       = 42

# Categories that need validation (excludes sentinels)
SKIP_LABELS = {"-", "NO_MATCH", "UNKNOWN"}

_SENTENCE_RE = re.compile(r"^Sentença", re.IGNORECASE)


# ------------------------------------------------------------------ #
# Parsing
# ------------------------------------------------------------------ #

def load_spans(tsv_path: str | Path) -> list[dict]:
    """
    Parse a 4-column BIO taxonomy TSV and return one dict per error span.

    Each dict has:
      sentence_id   : int (1-based)
      sentence_text : str (full sentence, space-joined tokens)
      span_start    : int (token index, inclusive)
      span_end      : int (token index, inclusive)
      span_text     : str (raw span tokens joined)
      highlighted   : str (full sentence with span in **asterisks**)
      taxonomy      : str (category from column 4)
    """
    spans = []
    sentence_id   = 0
    tokens:   list[str] = []
    labels:   list[str] = []
    taxonomy: list[str] = []

    def flush():
        if not tokens:
            return
        sentence_text = " ".join(tokens)
        i = 0
        while i < len(labels):
            if labels[i] == "B-WRONG":
                j = i + 1
                while j < len(labels) and labels[j] == "I-WRONG":
                    j += 1
                cat = taxonomy[i]
                if cat in SKIP_LABELS:
                    i = j
                    continue

                span_toks = tokens[i:j]
                span_text = " ".join(span_toks)

                # Build highlighted sentence: wrap span in **
                highlighted = (
                    " ".join(tokens[:i])
                    + (" " if i > 0 else "")
                    + "**" + span_text + "**"
                    + (" " if j < len(tokens) else "")
                    + " ".join(tokens[j:])
                ).strip()

                spans.append({
                    "sentence_id":   sentence_id,
                    "sentence_text": sentence_text,
                    "span_start":    i,
                    "span_end":      j - 1,
                    "span_text":     span_text,
                    "highlighted":   highlighted,
                    "taxonomy":      cat,
                })
                i = j
            else:
                i += 1

    with open(tsv_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if _SENTENCE_RE.match(line):
                flush()
                sentence_id += 1
                tokens, labels, taxonomy = [], [], []
            elif not line.strip():
                flush()
                tokens, labels, taxonomy = [], [], []
            else:
                parts = line.split("\t")
                if len(parts) >= 4:
                    tokens.append(parts[0])
                    labels.append(parts[1].strip())
                    taxonomy.append(parts[3].strip())
    flush()
    return spans


# ------------------------------------------------------------------ #
# Stratified sampling
# ------------------------------------------------------------------ #

def stratified_sample(
    spans: list[dict],
    total: int = TOTAL_SAMPLE,
    min_per_cat: int = MIN_PER_CATEGORY,
    seed: int = RANDOM_SEED,
) -> list[dict]:
    rng = random.Random(seed)

    # Group by category
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for s in spans:
        by_cat[s["taxonomy"]].append(s)

    categories = list(by_cat.keys())
    print(f"\nCategory counts in train (before sampling):")
    for cat in sorted(categories, key=lambda c: -len(by_cat[c])):
        print(f"  {cat:<35} {len(by_cat[cat]):>6} spans")
    print(f"  {'TOTAL':<35} {len(spans):>6} spans\n")

    # Phase 1: guarantee minimum per category
    selected: dict[str, list[dict]] = {}
    for cat in categories:
        pool = by_cat[cat][:]
        rng.shuffle(pool)
        selected[cat] = pool[:min_per_cat]

    already = sum(len(v) for v in selected.values())
    remaining = total - already

    if remaining <= 0:
        result = [s for v in selected.values() for s in v]
        rng.shuffle(result)
        return result

    # Phase 2: fill remaining slots proportionally
    # Only categories that still have unused spans
    total_available = sum(
        max(0, len(by_cat[c]) - len(selected[c])) for c in categories
    )
    for cat in categories:
        leftover = [s for s in by_cat[cat] if s not in selected[cat]]
        if not leftover or total_available == 0:
            continue
        proportion = len(by_cat[cat]) / sum(len(v) for v in by_cat.values())
        extra = round(remaining * proportion)
        extra = min(extra, len(leftover))
        selected[cat].extend(leftover[:extra])

    # Trim to total if rounding pushed us over
    result = [s for v in selected.values() for s in v]
    rng.shuffle(result)
    return result[:total]


# ------------------------------------------------------------------ #
# CSV output
# ------------------------------------------------------------------ #

FIELDNAMES = [
    "sentence_id",
    "full_sentence",
    "span_text",
    "taxonomy_auto",
    "is_error",          # linguist: yes / no / unsure
    "taxonomy_correct",  # linguist: leave blank if taxonomy_auto is correct
    "notes",
]


def write_csv(sample: list[dict], output_path: str | Path) -> None:
    # Sort by sentence_id for readability
    sample = sorted(sample, key=lambda s: (s["sentence_id"], s["span_start"]))

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for s in sample:
            writer.writerow({
                "sentence_id":     s["sentence_id"],
                "full_sentence":   s["highlighted"],
                "span_text":       s["span_text"],
                "taxonomy_auto":   s["taxonomy"],
                "is_error":        "",
                "taxonomy_correct": "",
                "notes":           "",
            })

    print(f"Written {len(sample)} rows → {output_path}")


# ------------------------------------------------------------------ #
# Summary
# ------------------------------------------------------------------ #

def print_sample_summary(sample: list[dict]) -> None:
    counts: dict[str, int] = defaultdict(int)
    for s in sample:
        counts[s["taxonomy"]] += 1
    print("Sample distribution:")
    for cat, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:<35} {n:>4}")
    print(f"  {'TOTAL':<35} {sum(counts.values()):>4}")


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample taxonomy spans from train for linguist validation"
    )
    parser.add_argument(
        "--input", default="data/train_bio_taxonomy.tsv",
        help="4-column taxonomy TSV (default: data/train_bio_taxonomy.tsv)"
    )
    parser.add_argument(
        "--output", default="data/taxonomy_validation_sample.csv",
        help="Output CSV path (default: data/taxonomy_validation_sample.csv)"
    )
    parser.add_argument(
        "--total", type=int, default=TOTAL_SAMPLE,
        help=f"Total sample size (default: {TOTAL_SAMPLE})"
    )
    parser.add_argument(
        "--min-per-cat", type=int, default=MIN_PER_CATEGORY,
        help=f"Minimum spans per category (default: {MIN_PER_CATEGORY})"
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED})"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"Loading spans from {args.input}...")
    spans = load_spans(args.input)
    print(f"Loaded {len(spans)} labelled spans.")

    sample = stratified_sample(
        spans,
        total=args.total,
        min_per_cat=args.min_per_cat,
        seed=args.seed,
    )

    print_sample_summary(sample)
    write_csv(sample, args.output)