"""
cogroo_taxonomy.py
Adds a fourth taxonomy column to CoGrOO-annotated BIO TSV files and
computes full statistics for error type analysis.

Input  (3 columns): token  BIO_label  cogroo_type
Output (4 columns): token  BIO_label  cogroo_type  taxonomy_category

Taxonomy categories:
  ortografia_lexical    : spelling, wrong word, lexical substitution
  ortografia_acento     : missing/wrong accent, accentuation confusion
  concordancia_verbal   : subject-verb agreement
  concordancia_nominal  : adjective-noun, article-noun agreement
  crase_preposicao      : crase, preposition use
  colocacao_pronominal  : pronoun placement
  sintatico_discursivo  : word order, discourse, repetition, style
  regencia              : verbal/nominal regency
  NO_MATCH              : CoGrOO flag with no gold annotation
  UNKNOWN               : gold error CoGrOO missed (heuristic applied)
  -                     : correct token

Usage:
    python cogroo_taxonomy.py
"""

from __future__ import annotations
import re
from collections import Counter, defaultdict
from pathlib import Path


# ------------------------------------------------------------------ #
# CoGrOO short_msg → taxonomy mapping
# ------------------------------------------------------------------ #

COGROO_TO_TAXONOMY: dict[str, str] = {
    # Concordância verbal
    "Verificou-se erro de concordância entre o sujeito e o verbo.": "concordancia_verbal",
    "O adjetivo na função de predicativo concorda com o verbo.": "concordancia_verbal",
    # Concordância nominal
    "O adjetivo concorda com o substantivo a que se refere.": "concordancia_nominal",
    "O adjetivo na função de predicativo concorda com o sujeito.": "concordancia_nominal",
    "Os artigos concordam com o substantivo a que se referem.": "concordancia_nominal",
    "Os determinantes concordam com o substantivo a que se referem.": "concordancia_nominal",
    "Os numerais concordam com o substantivo a que se referem.": "concordancia_nominal",
    # Crase e preposição
    "Não ocorre crase antes de palavras masculinas.": "crase_preposicao",
    "Não acontece crase antes de verbo.": "crase_preposicao",
    "Não há crase neste caso, somente no plural (\"às\").": "crase_preposicao",
    "Não há crase porque alguns pronomes pessoais não admitem artigo.": "crase_preposicao",
    "Alguns nomes regem a preposição \"a\", logo há crase aqui.": "crase_preposicao",
    "É inadequado o uso da preposição \"em\".": "crase_preposicao",
    "\"Em relação\" rege a preposição \"a\", logo há crase aqui.": "crase_preposicao",
    "\"Devido\" rege a preposição \"a\", logo há crase aqui.": "crase_preposicao",
    "(Des)obedecer\" constrói-se com prep. \"a\". Há crase com compl. feminino.": "crase_preposicao",
    # Colocação pronominal
    "Pron. relativos e conj. subordinativas atraem o pronome para antes do verbo.": "colocacao_pronominal",
    "Palavras negativas atraem o pronome átono para antes do verbo.": "colocacao_pronominal",
    "Certos advérbios atraem o pronome para antes do verbo.": "colocacao_pronominal",
    "Alguns pronomes indefinidos atraem o pronome para antes do verbo.": "colocacao_pronominal",
    # Regência
    "Regência verbal.": "regencia",
    "Os verbos \"evitar\" e \"usufruir\" não regem preposição \"de\".": "regencia",
    # Ortografia/acento
    "O contrário de \"bem\" é \"mal\", e o de \"bom\" é \"mau\".": "ortografia_lexical",
    # Sintático/discursivo
    "Repetição de palavras.": "sintatico_discursivo",
    "Deve haver vírgula antes de \"no entanto\".": "sintatico_discursivo",
    "A expressão \"ou seja\" deve ser isolada por vírgulas.": "sintatico_discursivo",
    "Verifique o excesso de verbos em sequência.": "sintatico_discursivo",
}

# Pattern for "Possível confusão entre X e X." → ortografia_acento
_CONFUSION_RE = re.compile(r"Possível confusão entre .+ e .+\.", re.IGNORECASE)


def cogroo_msg_to_taxonomy(msg: str) -> str:
    """Map a CoGrOO short_msg to a taxonomy category."""
    if msg in ("-", "UNKNOWN", "NO_MATCH"):
        return msg
    if msg in COGROO_TO_TAXONOMY:
        return COGROO_TO_TAXONOMY[msg]
    if _CONFUSION_RE.match(msg):
        return "ortografia_acento"
    # Unknown CoGrOO message — return as-is for inspection
    return f"cogroo_other:{msg[:40]}"


# ------------------------------------------------------------------ #
# Heuristic taxonomy for UNKNOWN spans
# ------------------------------------------------------------------ #

_VERBAL_RE = re.compile(
    r'\b(foi|foram|era|eram|é|são|está|estão|tem|têm|vem|vêm|'
    r'ser|estar|ter|haver|faz|fazem|vai|vão|deve|devem|pode|podem)\b',
    re.IGNORECASE
)
_NOMINAL_RE = re.compile(
    r'\b(os|as|uns|umas|maus|más|bons|boas|todos|todas|'
    r'muitos|muitas|poucos|poucas|outros|outras)\b',
    re.IGNORECASE
)
_CRASE_RE = re.compile(
    r'^(a|as|ao|aos|à|às)$',
    re.IGNORECASE
)
_ACCENT_RE = re.compile(r'[àáâãéêíóôõúç]')
_PRONOUN_RE = re.compile(
    r'\b(se|me|te|lhe|nos|vos|lhes|o|a|os|as)\b',
    re.IGNORECASE
)


def classify_unknown_span(span_tokens: list[str]) -> str:
    """Heuristic classification of UNKNOWN error spans."""
    text = " ".join(span_tokens)

    # Single token cases
    if len(span_tokens) == 1:
        token = span_tokens[0]
        if _CRASE_RE.match(token):
            return "crase_preposicao"
        if _ACCENT_RE.search(token):
            return "ortografia_acento"
        return "ortografia_lexical"

    # Multi-token: check for verbal agreement
    if _VERBAL_RE.search(text):
        return "concordancia_verbal"

    # Nominal agreement
    if _NOMINAL_RE.search(text):
        return "concordancia_nominal"

    # Pronoun placement
    if _PRONOUN_RE.search(text) and len(span_tokens) <= 3:
        return "colocacao_pronominal"

    # Longer spans → syntactic/discourse
    if len(span_tokens) >= 3:
        return "sintatico_discursivo"

    return "ortografia_lexical"


# ------------------------------------------------------------------ #
# File processing
# ------------------------------------------------------------------ #

_SENTENCE_RE = re.compile(r"^Sentença", re.IGNORECASE)


def add_taxonomy_column(input_path: str | Path, output_path: str | Path) -> None:
    """
    Read a 3-column BIO TSV and write a 4-column version
    with the taxonomy category added as the fourth column.
    """
    # First pass: read all sentences
    sentences: list[tuple[int, list[str], list[str], list[str]]] = []
    sentence_id = 0
    current_tokens: list[str] = []
    current_labels: list[str] = []
    current_types: list[str] = []

    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if _SENTENCE_RE.match(line):
                if current_tokens:
                    sentences.append((sentence_id, current_tokens, current_labels, current_types))
                sentence_id += 1
                current_tokens, current_labels, current_types = [], [], []
            elif not line.strip():
                if current_tokens:
                    sentences.append((sentence_id, current_tokens, current_labels, current_types))
                    current_tokens, current_labels, current_types = [], [], []
            else:
                parts = line.split("\t")
                if len(parts) >= 3:
                    current_tokens.append(parts[0])
                    current_labels.append(parts[1].strip())
                    current_types.append(parts[2].strip())
                elif len(parts) == 2:
                    current_tokens.append(parts[0])
                    current_labels.append(parts[1].strip())
                    current_types.append("-")
    if current_tokens:
        sentences.append((sentence_id, current_tokens, current_labels, current_types))

    # Compute taxonomy per token
    sent_taxonomy: dict[int, list[str]] = {}
    for sid, tokens, labels, types in sentences:
        taxonomy = ["-"] * len(tokens)

        # Process span by span
        i = 0
        while i < len(labels):
            if labels[i] == "B-WRONG":
                # Find span end
                j = i + 1
                while j < len(labels) and labels[j] == "I-WRONG":
                    j += 1
                span_type = types[i]  # all tokens in span share B-WRONG token's type

                if span_type == "UNKNOWN":
                    cat = classify_unknown_span(tokens[i:j])
                elif span_type == "NO_MATCH":
                    cat = "NO_MATCH"
                elif span_type == "-":
                    cat = "-"
                else:
                    cat = cogroo_msg_to_taxonomy(span_type)

                for k in range(i, j):
                    taxonomy[k] = cat
                i = j
            elif labels[i] == "O":
                # Check for NO_MATCH on O tokens
                if types[i] == "NO_MATCH":
                    taxonomy[i] = "NO_MATCH"
                i += 1
            else:
                i += 1

        sent_taxonomy[sid] = taxonomy

    # Second pass: write output
    sentence_id = 0
    tok_idx = 0
    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.rstrip("\n")
            if _SENTENCE_RE.match(line):
                sentence_id += 1
                tok_idx = 0
                fout.write(line + "\n")
            elif not line.strip():
                fout.write("\n")
            else:
                parts = line.split("\t")
                if len(parts) >= 3:
                    token, label, cogroo_type = parts[0], parts[1], parts[2]
                    tax = sent_taxonomy.get(sentence_id, [])
                    taxonomy_cat = tax[tok_idx] if tok_idx < len(tax) else "-"
                    fout.write(f"{token}\t{label}\t{cogroo_type}\t{taxonomy_cat}\n")
                    tok_idx += 1
                else:
                    fout.write(line + "\n")

    print(f"Written: {output_path}")


# ------------------------------------------------------------------ #
# Statistics
# ------------------------------------------------------------------ #

def compute_statistics(typed_path: str | Path) -> dict:
    """
    Compute full error type statistics from a 4-column BIO TSV file.
    Returns a dict with span-level and token-level counts.
    """
    span_taxonomy_counts: Counter = Counter()
    token_taxonomy_counts: Counter = Counter()
    cogroo_raw_counts: Counter = Counter()
    no_match_spans = 0
    total_gold_spans = 0
    total_tokens = 0

    sentence_id = 0
    current_tokens: list[str] = []
    current_labels: list[str] = []
    current_types: list[str] = []
    current_taxonomy: list[str] = []

    def flush():
        nonlocal no_match_spans, total_gold_spans, total_tokens
        nonlocal span_taxonomy_counts, token_taxonomy_counts, cogroo_raw_counts

        total_tokens += len(current_tokens)

        # Token-level counts
        for lbl, cogroo, tax in zip(current_labels, current_types, current_taxonomy):
            token_taxonomy_counts[tax] += 1
            if cogroo not in ("-", "UNKNOWN", "NO_MATCH"):
                cogroo_raw_counts[cogroo] += 1

        # Span-level counts
        i = 0
        while i < len(current_labels):
            if current_labels[i] == "B-WRONG":
                j = i + 1
                while j < len(current_labels) and current_labels[j] == "I-WRONG":
                    j += 1
                total_gold_spans += 1
                span_taxonomy_counts[current_taxonomy[i]] += 1
                i = j
            else:
                if current_types[i] == "NO_MATCH" and current_taxonomy[i] == "NO_MATCH":
                    no_match_spans += 1
                i += 1

    with open(typed_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if _SENTENCE_RE.match(line):
                if current_tokens:
                    flush()
                current_tokens, current_labels = [], []
                current_types, current_taxonomy = [], []
                sentence_id += 1
            elif not line.strip():
                if current_tokens:
                    flush()
                current_tokens, current_labels = [], []
                current_types, current_taxonomy = [], []
            else:
                parts = line.split("\t")
                if len(parts) >= 4:
                    current_tokens.append(parts[0])
                    current_labels.append(parts[1].strip())
                    current_types.append(parts[2].strip())
                    current_taxonomy.append(parts[3].strip())
    if current_tokens:
        flush()

    cogroo_matched = sum(
        v for k, v in span_taxonomy_counts.items()
        if k not in ("UNKNOWN", "NO_MATCH", "-")
    )
    unknown_spans = span_taxonomy_counts.get("UNKNOWN", 0) + sum(
        v for k, v in span_taxonomy_counts.items()
        if k not in ("UNKNOWN", "NO_MATCH", "-") and k != "UNKNOWN"
    )

    return {
        "total_tokens": total_tokens,
        "total_gold_spans": total_gold_spans,
        "cogroo_matched_spans": cogroo_matched,
        "unknown_spans": span_taxonomy_counts.get("UNKNOWN", 0),
        "no_match_spans": no_match_spans,
        "span_taxonomy": dict(span_taxonomy_counts),
        "token_taxonomy": dict(token_taxonomy_counts),
        "cogroo_raw": dict(cogroo_raw_counts),
    }


def print_statistics(stats: dict, split_name: str) -> None:
    print(f"\n{'='*60}")
    print(f"  Split: {split_name}")
    print(f"{'='*60}")
    print(f"  Total tokens         : {stats['total_tokens']:>8}")
    print(f"  Total gold spans     : {stats['total_gold_spans']:>8}")
    print(f"  CoGrOO matched spans : {stats['cogroo_matched_spans']:>8} "
          f"({100*stats['cogroo_matched_spans']/max(stats['total_gold_spans'],1):.1f}%)")
    print(f"  UNKNOWN spans        : {stats['unknown_spans']:>8} "
          f"({100*stats['unknown_spans']/max(stats['total_gold_spans'],1):.1f}%)")
    print(f"  NO_MATCH spans       : {stats['no_match_spans']:>8}")

    print(f"\n  --- Span-level taxonomy distribution ---")
    total_spans = sum(stats['span_taxonomy'].values())
    for cat, count in sorted(stats['span_taxonomy'].items(), key=lambda x: -x[1]):
        pct = 100 * count / total_spans if total_spans else 0
        print(f"    {cat:<35} {count:>6} ({pct:.1f}%)")

    print(f"\n  --- CoGrOO raw categories (matched spans only) ---")
    for msg, count in sorted(stats['cogroo_raw'].items(), key=lambda x: -x[1]):
        print(f"    {msg:<60} {count:>5}")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    splits = ["train", "val", "test"]
    all_stats: dict[str, dict] = {}

    for split in splits:
        input_path = Path(f"data/{split}_bio_typed.tsv")
        output_path = Path(f"data/{split}_bio_taxonomy.tsv")

        if not input_path.exists():
            print(f"Skipping {split} — {input_path} not found")
            continue

        print(f"\nProcessing {split}...")
        add_taxonomy_column(input_path, output_path)
        stats = compute_statistics(output_path)
        all_stats[split] = stats
        print_statistics(stats, split)

    # Cross-split summary table
    if len(all_stats) > 1:
        print(f"\n{'='*60}")
        print("  Cross-split taxonomy summary (span %)")
        print(f"{'='*60}")
        all_cats = sorted(set(
            cat for s in all_stats.values()
            for cat in s['span_taxonomy'].keys()
        ))
        header = f"{'Category':<35}" + "".join(f"{s:>10}" for s in all_stats.keys())
        print(f"  {header}")
        print(f"  {'-'*60}")
        for cat in all_cats:
            row = f"  {cat:<35}"
            for split, stats in all_stats.items():
                total = sum(stats['span_taxonomy'].values())
                count = stats['span_taxonomy'].get(cat, 0)
                pct = 100 * count / total if total else 0
                row += f"{pct:>9.1f}%"
            print(row)