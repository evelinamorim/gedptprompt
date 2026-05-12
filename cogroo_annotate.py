"""
cogroo_annotate.py
Runs CoGrOO on BIO-tagged TSV files and produces an enriched output
with a third column containing the error type for each token.

Third column values:
  -          : token is correct (O label) and CoGrOO found no error
  <type>     : token is part of a gold error span matched by CoGrOO (overlap >= 50%)
  UNKNOWN    : token is part of a gold error span that CoGrOO did NOT catch
  NO_MATCH   : token has no gold error but CoGrOO flagged it as wrong

Input format (BIO TSV):
    Sentença Original N°1
    O       O
    maior   O
    ...

Output format (BIO TSV + error type):
    Sentença Original N°1
    O       O       -
    maior   O       -
    Entre   B-WRONG concordância_verbal
    muitos  I-WRONG concordância_verbal

Usage:
    python cogroo_annotate.py --input data/test_bio.tsv --output data/test_bio_typed.tsv
    python cogroo_annotate.py --input data/train_bio.tsv --output data/train_bio_typed.tsv
    python cogroo_annotate.py --input data/val_bio.tsv --output data/val_bio_typed.tsv
"""

from __future__ import annotations
import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

try:
    ifrom cogroo4py.cogroo import Cogroo
except ImportError:
    print("ERROR: cogroo4py not installed. Run: pip install cogroo4py")
    sys.exit(1)

from data_reader import Sentence, read_bio_file


# ------------------------------------------------------------------ #
# CoGrOO wrapper
# ------------------------------------------------------------------ #
class CoGrOOChecker:
    def __init__(self):
        print("Initializing CoGrOO (this may take a few seconds)...")
        self.checker = Cogroo()
        print("CoGrOO ready.")

    def check(self, text: str) -> list[dict]:
        mistakes = []
        try:
            doc = self.checker.grammar_check(text)
            for mistake in doc.mistakes:
                mistakes.append({
                    "start": mistake.start,
                    "end": mistake.end,
                    "rule_id": getattr(mistake, "rule_id", ""),
                    "category": getattr(mistake, "category", "grammar_error"),
                    "short_msg": getattr(mistake, "short_msg", ""),
                })
        except Exception as e:
            print(f"  [WARNING] CoGrOO error on text '{text[:50]}': {e}")
        return mistakes


# ------------------------------------------------------------------ #
# Token offset computation
# ------------------------------------------------------------------ #

def compute_token_offsets(tokens: list[str]) -> list[tuple[int, int]]:
    """
    Compute character offsets for each token assuming single-space joining.
    Returns list of (start, end_exclusive) tuples.
    """
    offsets = []
    pos = 0
    for token in tokens:
        start = pos
        end = pos + len(token)
        offsets.append((start, end))
        pos = end + 1  # +1 for space separator
    return offsets


# ------------------------------------------------------------------ #
# Span overlap computation
# ------------------------------------------------------------------ #

def compute_overlap_ratio(
    cogroo_start: int,
    cogroo_end: int,
    gold_token_offsets: list[tuple[int, int]],
    gold_span_start_idx: int,
    gold_span_end_idx: int,
) -> float:
    """
    Compute overlap ratio between a CoGrOO character span and a gold token span.
    Returns the fraction of the gold span covered by the CoGrOO span.
    """
    gold_char_start = gold_token_offsets[gold_span_start_idx][0]
    gold_char_end = gold_token_offsets[gold_span_end_idx][1]
    gold_len = gold_char_end - gold_char_start

    if gold_len == 0:
        return 0.0

    overlap_start = max(cogroo_start, gold_char_start)
    overlap_end = min(cogroo_end, gold_char_end)
    overlap_len = max(0, overlap_end - overlap_start)

    return overlap_len / gold_len


# ------------------------------------------------------------------ #
# Gold span extraction
# ------------------------------------------------------------------ #

@dataclass
class GoldSpan:
    start_idx: int   # token index
    end_idx: int     # token index inclusive
    error_type: str = "UNKNOWN"   # filled in after CoGrOO matching


def extract_gold_spans(labels: list[str]) -> list[GoldSpan]:
    """Extract gold error spans from BIO labels."""
    spans = []
    start = None
    for i, lbl in enumerate(labels):
        if lbl == "B-WRONG":
            if start is not None:
                spans.append(GoldSpan(start, i - 1))
            start = i
        elif lbl == "O" and start is not None:
            spans.append(GoldSpan(start, i - 1))
            start = None
    if start is not None:
        spans.append(GoldSpan(start, len(labels) - 1))
    return spans


# ------------------------------------------------------------------ #
# Main matching logic
# ------------------------------------------------------------------ #

def match_cogroo_to_gold(
    sentence: Sentence,
    cogroo_mistakes: list[dict],
    overlap_threshold: float = 0.5,
) -> list[str]:
    """
    Match CoGrOO mistakes to gold spans and return a list of error type
    strings, one per token.

    Token labels:
      "-"       : correct token, no CoGrOO flag
      <type>    : matched gold span with CoGrOO type
      "UNKNOWN" : gold error span not caught by CoGrOO
      "NO_MATCH": CoGrOO flagged but no gold span
    """
    token_types = ["-"] * len(sentence.tokens)
    token_offsets = compute_token_offsets(sentence.tokens)
    gold_spans = extract_gold_spans(sentence.labels)

    # Track which CoGrOO mistakes have been matched
    matched_cogroo: set[int] = set()
    # Track which gold spans have been matched
    matched_gold: set[int] = set()

    # For each gold span, find the best overlapping CoGrOO mistake
    for gi, gold_span in enumerate(gold_spans):
        best_overlap = 0.0
        best_mistake_idx = -1
        best_category = "UNKNOWN"

        for mi, mistake in enumerate(cogroo_mistakes):
            overlap = compute_overlap_ratio(
                mistake["start"],
                mistake["end"],
                token_offsets,
                gold_span.start_idx,
                gold_span.end_idx,
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_mistake_idx = mi
                best_category = mistake["category"] or mistake["rule_id"] or "grammar_error"

        if best_overlap >= overlap_threshold:
            # Gold span matched by CoGrOO
            matched_cogroo.add(best_mistake_idx)
            matched_gold.add(gi)
            for idx in range(gold_span.start_idx, gold_span.end_idx + 1):
                token_types[idx] = best_category
        else:
            # Gold span not caught by CoGrOO
            for idx in range(gold_span.start_idx, gold_span.end_idx + 1):
                token_types[idx] = "UNKNOWN"

    # Handle CoGrOO mistakes with no gold match → NO_MATCH
    for mi, mistake in enumerate(cogroo_mistakes):
        if mi in matched_cogroo:
            continue
        # Find tokens covered by this CoGrOO mistake
        for ti, (tok_start, tok_end) in enumerate(token_offsets):
            if tok_start < mistake["end"] and tok_end > mistake["start"]:
                # Only mark as NO_MATCH if not already assigned a gold type
                if token_types[ti] == "-":
                    token_types[ti] = "NO_MATCH"

    return token_types


# ------------------------------------------------------------------ #
# File writing
# ------------------------------------------------------------------ #

_SENTENCE_RE = re.compile(r"^Sentença", re.IGNORECASE)


def write_typed_bio(
    input_path: str | Path,
    output_path: str | Path,
    checker: CoGrOOChecker,
    overlap_threshold: float = 0.5,
    progress_every: int = 100,
) -> None:
    """
    Read a BIO TSV file, annotate with CoGrOO error types,
    and write the enriched output with a third column.
    """
    sentences = read_bio_file(input_path)
    print(f"Loaded {len(sentences)} sentences from {input_path}")

    # Build sentence lookup by id
    sentence_map = {s.id: s for s in sentences}

    # Run CoGrOO on all sentences and collect results
    print("Running CoGrOO...")
    cogroo_results: dict[int, list[str]] = {}

    for i, sent in enumerate(sentences, 1):
        if i % progress_every == 0 or i == 1:
            print(f"  [{i}/{len(sentences)}] ...")

        text = " ".join(sent.tokens)
        mistakes = checker.check(text)
        token_types = match_cogroo_to_gold(sent, mistakes, overlap_threshold)
        cogroo_results[sent.id] = token_types

    # Write output preserving original file structure
    print(f"Writing output to {output_path}...")
    sentence_idx = 0

    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        current_token_idx = 0

        for raw_line in fin:
            line = raw_line.rstrip("\n")

            # Sentence boundary line
            if _SENTENCE_RE.match(line):
                sentence_idx += 1
                current_token_idx = 0
                fout.write(line + "\n")
                continue

            # Blank line
            if not line.strip():
                fout.write("\n")
                continue

            # Token line
            parts = line.split("\t")
            if len(parts) < 2:
                parts = line.split()
            if len(parts) < 2:
                fout.write(line + "\n")
                continue

            token, label = parts[0], parts[1].strip()

            # Get error type for this token
            if sentence_idx in cogroo_results:
                types = cogroo_results[sentence_idx]
                if current_token_idx < len(types):
                    error_type = types[current_token_idx]
                else:
                    error_type = "-"
            else:
                error_type = "-"

            fout.write(f"{token}\t{label}\t{error_type}\n")
            current_token_idx += 1

    print(f"Done. Output written to {output_path}")


# ------------------------------------------------------------------ #
# Summary statistics
# ------------------------------------------------------------------ #

def print_summary(output_path: str | Path) -> None:
    """Print a summary of error type distribution in the output file."""
    from collections import Counter
    type_counts: Counter = Counter()

    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or _SENTENCE_RE.match(line):
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                error_type = parts[2].strip()
                type_counts[error_type] += 1

    total = sum(type_counts.values())
    print(f"\n=== Error type distribution ({output_path}) ===")
    print(f"{'Type':<30} {'Count':>8} {'%':>8}")
    print("-" * 50)
    for etype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total if total else 0
        print(f"{etype:<30} {count:>8} {pct:>7.2f}%")
    print(f"{'TOTAL':<30} {total:>8}")


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate BIO TSV files with CoGrOO error types"
    )
    parser.add_argument(
        "--input", required=True,
        help="Input BIO TSV file (e.g. data/test_bio.tsv)"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output BIO TSV file with error type column"
    )
    parser.add_argument(
        "--overlap", type=float, default=0.5,
        help="Minimum overlap ratio to match CoGrOO span to gold span (default: 0.5)"
    )
    parser.add_argument(
        "--progress", type=int, default=100,
        help="Print progress every N sentences (default: 100)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    checker = CoGrOOChecker()

    write_typed_bio(
        input_path=args.input,
        output_path=args.output,
        checker=checker,
        overlap_threshold=args.overlap,
        progress_every=args.progress,
    )

    print_summary(args.output)