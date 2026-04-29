"""
data_reader.py
Reads BIO-tagged TSV files for the GED-PT benchmark.

File format:
    Sentença Original N°1          <- sentence boundary marker
    O       O                      <- token <TAB> label
    maior   O
    ...
    Sentença Original N°2
    Entre   B-WRONG
    muitos  I-WRONG
    ...
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class Sentence:
    id: int                        # 1-based sentence index from the file
    tokens: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.tokens)

    def text(self, sep: str = " ") -> str:
        return sep.join(self.tokens)

    def error_spans(self) -> list[tuple[int, int]]:
        """Return list of (start, end_inclusive) token indices for each error span."""
        spans: list[tuple[int, int]] = []
        start: int | None = None
        for i, lbl in enumerate(self.labels):
            if lbl == "B-WRONG":
                start = i
            elif lbl == "O" and start is not None:
                spans.append((start, i - 1))
                start = None
        if start is not None:
            spans.append((start, len(self.labels) - 1))
        return spans

    def __repr__(self) -> str:
        return f"Sentence(id={self.id}, tokens={self.tokens}, labels={self.labels})"


# ------------------------------------------------------------------ #
_SENTENCE_RE = re.compile(r"^Sentença", re.IGNORECASE)


def read_bio_file(path: str | Path, *, valid_labels: set[str] | None = None) -> list[Sentence]:
    """
    Parse a BIO TSV file and return a list of Sentence objects.

    Parameters
    ----------
    path:
        Path to the .tsv file.
    valid_labels:
        If provided, labels not in this set are replaced with 'O' and a
        warning is printed (handles minor model hallucinations when reusing
        the reader on predicted files).

    Returns
    -------
    list[Sentence]
    """
    if valid_labels is None:
        valid_labels = {"O", "B-WRONG", "I-WRONG"}

    sentences: list[Sentence] = []
    current: Sentence | None = None
    sentence_idx = 0

    with open(path, encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")

            # --- sentence boundary ---
            if _SENTENCE_RE.match(line):
                if current is not None and current.tokens:
                    sentences.append(current)
                sentence_idx += 1
                current = Sentence(id=sentence_idx)
                continue

            # --- blank line: also treated as boundary ---
            if not line.strip():
                if current is not None and current.tokens:
                    sentences.append(current)
                    current = None
                continue

            # --- token line ---
            parts = line.split("\t")
            if len(parts) < 2:
                # fallback: split on any whitespace
                parts = line.split()
            if len(parts) < 2:
                # single-column line — skip silently
                continue

            token, label = parts[0], parts[1].strip()
            if label not in valid_labels:
                print(f"[WARNING] Unknown label '{label}' at sentence {sentence_idx}, "
                      f"token '{token}' — replaced with 'O'.")
                label = "O"

            if current is None:
                # file started without a sentence marker
                sentence_idx += 1
                current = Sentence(id=sentence_idx)
            current.tokens.append(token)
            current.labels.append(label)

    # flush last sentence
    if current is not None and current.tokens:
        sentences.append(current)

    return sentences


# ------------------------------------------------------------------ #
def iter_bio_file(path: str | Path) -> Iterator[Sentence]:
    """Memory-efficient generator version of read_bio_file."""
    yield from read_bio_file(path)


# ------------------------------------------------------------------ #
def split_summary(sentences: list[Sentence]) -> dict:
    """Print a quick summary of the dataset split."""
    total_tokens = sum(len(s) for s in sentences)
    total_error_spans = sum(len(s.error_spans()) for s in sentences)
    error_tokens = sum(
        sum(1 for lbl in s.labels if lbl != "O") for s in sentences
    )
    return {
        "num_sentences": len(sentences),
        "num_tokens": total_tokens,
        "num_error_spans": total_error_spans,
        "num_error_tokens": error_tokens,
        "pct_error_tokens": round(100 * error_tokens / total_tokens, 2) if total_tokens else 0,
    }


# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import sys
    import json

    path = sys.argv[1] if len(sys.argv) > 1 else "data/test_bio.tsv"
    sents = read_bio_file(path)
    summary = split_summary(sents)
    print(json.dumps(summary, indent=2))

    print("\nFirst 3 sentences:")
    for s in sents[:3]:
        print(f"  [{s.id}] tokens={s.tokens}")
        print(f"       labels={s.labels}")
        print(f"       spans ={s.error_spans()}")
