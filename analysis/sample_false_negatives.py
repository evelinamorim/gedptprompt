"""
sample_false_negatives.py

Samples N false negative spans from a GED prediction JSON,
stratified by error category, and exports to CSV for manual error analysis.

Taxonomy is generated on the fly from the gold BIO file — no pre-existing
taxonomy.tsv needed. The gold file can be either:
  - 2 columns: token  BIO_label           (UNKNOWN heuristic applied to all)
  - 3 columns: token  BIO_label  cogroo_type  (full taxonomy mapping)
  - 4 columns: token  BIO_label  cogroo_type  taxonomy_category  (already done)

Usage:
    python sample_false_negatives.py \
        --predictions predictions/Qwen-Qwen3-8B_test_two_stage.json \
        --gold data/test_bio_typed.tsv \
        --output fn_sample_300.csv \
        --n 300 \
        --seed 42
"""

import json
import csv
import argparse
import random
import re
from collections import defaultdict
from math import floor


# ============================================================
# Taxonomy logic (from cogroo_taxonomy.py)
# ============================================================

COGROO_TO_TAXONOMY = {
    "Verificou-se erro de concordância entre o sujeito e o verbo.": "concordancia_verbal",
    "O adjetivo na função de predicativo concorda com o verbo.": "concordancia_verbal",
    "O adjetivo concorda com o substantivo a que se refere.": "concordancia_nominal",
    "O adjetivo na função de predicativo concorda com o sujeito.": "concordancia_nominal",
    "Os artigos concordam com o substantivo a que se referem.": "concordancia_nominal",
    "Os determinantes concordam com o substantivo a que se referem.": "concordancia_nominal",
    "Os numerais concordam com o substantivo a que se referem.": "concordancia_nominal",
    "Não ocorre crase antes de palavras masculinas.": "crase_preposicao",
    "Não acontece crase antes de verbo.": "crase_preposicao",
    "Não há crase neste caso, somente no plural (\"às\").": "crase_preposicao",
    "Não há crase porque alguns pronomes pessoais não admitem artigo.": "crase_preposicao",
    "Alguns nomes regem a preposição \"a\", logo há crase aqui.": "crase_preposicao",
    "É inadequado o uso da preposição \"em\".": "crase_preposicao",
    "\"Em relação\" rege a preposição \"a\", logo há crase aqui.": "crase_preposicao",
    "\"Devido\" rege a preposição \"a\", logo há crase aqui.": "crase_preposicao",
    "(Des)obedecer\" constrói-se com prep. \"a\". Há crase com compl. feminino.": "crase_preposicao",
    "Pron. relativos e conj. subordinativas atraem o pronome para antes do verbo.": "colocacao_pronominal",
    "Palavras negativas atraem o pronome átono para antes do verbo.": "colocacao_pronominal",
    "Certos advérbios atraem o pronome para antes do verbo.": "colocacao_pronominal",
    "Alguns pronomes indefinidos atraem o pronome para antes do verbo.": "colocacao_pronominal",
    "Regência verbal.": "regencia",
    "Os verbos \"evitar\" e \"usufruir\" não regem preposição \"de\".": "regencia",
    "O contrário de \"bem\" é \"mal\", e o de \"bom\" é \"mau\".": "ortografia_lexical",
    "Repetição de palavras.": "sintatico_discursivo",
    "Deve haver vírgula antes de \"no entanto\".": "sintatico_discursivo",
    "A expressão \"ou seja\" deve ser isolada por vírgulas.": "sintatico_discursivo",
    "Verifique o excesso de verbos em sequência.": "sintatico_discursivo",
}

_CONFUSION_RE = re.compile(r"Possível confusão entre .+ e .+\.", re.IGNORECASE)
_SENTENCE_RE  = re.compile(r"^Sentença", re.IGNORECASE)

_VERBAL_RE  = re.compile(
    r'\b(foi|foram|era|eram|é|são|está|estão|tem|têm|vem|vêm|'
    r'ser|estar|ter|haver|faz|fazem|vai|vão|deve|devem|pode|podem)\b',
    re.IGNORECASE
)
_NOMINAL_RE = re.compile(
    r'\b(os|as|uns|umas|maus|más|bons|boas|todos|todas|'
    r'muitos|muitas|poucos|poucas|outros|outras)\b',
    re.IGNORECASE
)
_CRASE_RE   = re.compile(r'^(a|as|ao|aos|à|às)$', re.IGNORECASE)
_ACCENT_RE  = re.compile(r'[àáâãéêíóôõúç]')
_PRONOUN_RE = re.compile(
    r'\b(se|me|te|lhe|nos|vos|lhes|o|a|os|as)\b', re.IGNORECASE
)


def cogroo_msg_to_taxonomy(msg):
    if msg in ("-", "UNKNOWN", "NO_MATCH"):
        return msg
    if msg in COGROO_TO_TAXONOMY:
        return COGROO_TO_TAXONOMY[msg]
    if _CONFUSION_RE.match(msg):
        return "ortografia_acento"
    return "UNKNOWN"


def classify_unknown_span(span_tokens):
    text = " ".join(span_tokens)
    if len(span_tokens) == 1:
        token = span_tokens[0]
        if _CRASE_RE.match(token):
            return "crase_preposicao"
        if _ACCENT_RE.search(token):
            return "ortografia_acento"
        return "ortografia_lexical"
    if _VERBAL_RE.search(text):
        return "concordancia_verbal"
    if _NOMINAL_RE.search(text):
        return "concordancia_nominal"
    if _PRONOUN_RE.search(text) and len(span_tokens) <= 3:
        return "colocacao_pronominal"
    if len(span_tokens) >= 3:
        return "sintatico_discursivo"
    return "ortografia_lexical"


# ============================================================
# Gold file parsing → span taxonomy map
# ============================================================

def parse_gold_file(gold_path):
    """
    Parse a 2-, 3-, or 4-column BIO TSV and return:
        span_taxonomy:   dict[(sentence_id, span_start, span_end)] -> category
        sentence_tokens: dict[sentence_id] -> list[str]
    """
    span_taxonomy   = {}
    sentence_tokens = {}
    sentence_id = 0
    tokens, labels, types = [], [], []

    def flush():
        if not tokens:
            return
        i = 0
        while i < len(labels):
            if labels[i] == "B-WRONG":
                j = i + 1
                while j < len(labels) and labels[j] == "I-WRONG":
                    j += 1
                span_type = types[i] if types else "UNKNOWN"
                if span_type == "UNKNOWN":
                    cat = classify_unknown_span(tokens[i:j])
                elif span_type in ("-", "NO_MATCH"):
                    cat = span_type
                else:
                    cat = cogroo_msg_to_taxonomy(span_type)
                span_taxonomy[(sentence_id, i, j - 1)] = cat
                i = j
            else:
                i += 1
        sentence_tokens[sentence_id] = list(tokens)

    with open(gold_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if _SENTENCE_RE.match(line):
                flush()
                sentence_id += 1
                tokens.clear(); labels.clear(); types.clear()
            elif not line.strip():
                flush()
                tokens.clear(); labels.clear(); types.clear()
            else:
                parts = line.split("\t")
                if len(parts) >= 4:
                    # 4-col: taxonomy already computed, use col 4 directly
                    tokens.append(parts[0])
                    labels.append(parts[1].strip())
                    types.append(parts[3].strip())
                elif len(parts) == 3:
                    tokens.append(parts[0])
                    labels.append(parts[1].strip())
                    types.append(parts[2].strip())
                elif len(parts) == 2:
                    tokens.append(parts[0])
                    labels.append(parts[1].strip())
                    types.append("UNKNOWN")
    flush()
    return span_taxonomy, sentence_tokens


# ============================================================
# BIO helpers
# ============================================================

def bio_to_spans(labels):
    spans = []
    start = None
    for i, label in enumerate(labels):
        if label == "B-WRONG":
            if start is not None:
                spans.append((start, i - 1))
            start = i
        elif label == "I-WRONG":
            if start is None:
                start = i
        else:
            if start is not None:
                spans.append((start, i - 1))
                start = None
    if start is not None:
        spans.append((start, len(labels) - 1))
    return spans


# ============================================================
# False negative extraction
# ============================================================

def extract_false_negatives(predictions, span_taxonomy):
    false_negatives = []
    for entry in predictions:
        sid        = entry["sentence_id"]
        tokens     = entry["tokens"]
        gold_spans = set(map(tuple, bio_to_spans(entry["gold_labels"])))
        pred_spans = set(map(tuple, bio_to_spans(entry["pred_labels"])))

        for (start, end) in gold_spans - pred_spans:
            cat = span_taxonomy.get((sid, start, end), "UNKNOWN")
            false_negatives.append({
                "sentence_id":  sid,
                "span_start":   start,
                "span_end":     end,
                "span_tokens":  " ".join(tokens[start: end + 1]),
                "category":     cat,
                "context":      " ".join(tokens),
                "failure_mode": "",
                "notes":        "",
            })
    return false_negatives


# ============================================================
# Stratified sampling
# ============================================================

def stratified_sample(false_negatives, n, seed=42):
    rng = random.Random(seed)

    by_category = defaultdict(list)
    for fn in false_negatives:
        by_category[fn["category"]].append(fn)

    total = len(false_negatives)
    if total == 0:
        raise ValueError(
            "No false negatives found — check that predictions "
            "and gold file sentence IDs align."
        )
    if n >= total:
        print(f"[warn] Requested {n} but only {total} FNs exist — returning all.")
        return list(false_negatives)

    # Proportional quotas, minimum 1 per category
    quotas = {
        cat: max(1, floor(n * len(items) / total))
        for cat, items in by_category.items()
    }

    # Fix rounding so total == n
    diff = n - sum(quotas.values())
    for cat in sorted(by_category, key=lambda c: -len(by_category[c])):
        if diff == 0:
            break
        quotas[cat] += 1
        diff -= 1

    sampled = []
    for cat, items in by_category.items():
        k = min(quotas[cat], len(items))
        sampled.extend(rng.sample(items, k))

    rng.shuffle(sampled)
    return sampled


# ============================================================
# Summary + CSV export
# ============================================================

FIELDNAMES = [
    "sentence_id", "span_start", "span_end",
    "span_tokens", "category",
    "failure_mode", "notes",
    "context",
]

FAILURE_MODES = "complete_miss | partial_span | boundary_shift | wrong_token"


def print_summary(false_negatives, sampled):
    total = len(false_negatives)
    print(f"\nTotal false negatives : {total}")
    print(f"Sampled               : {len(sampled)}\n")

    by_total   = defaultdict(int)
    by_sampled = defaultdict(int)
    for fn in false_negatives:
        by_total[fn["category"]] += 1
    for fn in sampled:
        by_sampled[fn["category"]] += 1

    print(f"{'Category':<30} {'Total FNs':>10} {'Sampled':>10} {'% of FNs':>10}")
    print("-" * 64)
    for cat in sorted(by_total, key=lambda c: -by_total[c]):
        t   = by_total[cat]
        s   = by_sampled.get(cat, 0)
        pct = 100 * t / total
        print(f"{cat:<30} {t:>10} {s:>10} {pct:>9.1f}%")


def export_csv(sampled, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        # Instruction row for annotator
        writer.writerow({
            "sentence_id":  "# failure_mode values:",
            "span_start":   FAILURE_MODES,
            "span_end":     "",
            "span_tokens":  "",
            "category":     "",
            "failure_mode": "",
            "notes":        "",
            "context":      "",
        })
        for row in sampled:
            writer.writerow({k: row.get(k, "") for k in FIELDNAMES})
    print(f"Saved {len(sampled)} rows → {output_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Sample false negatives for error analysis, "
                    "with on-the-fly taxonomy generation."
    )
    parser.add_argument("--predictions", required=True,
                        help="Prediction JSON (e.g. Qwen-Qwen3-8B_test_two_stage.json)")
    parser.add_argument("--gold", required=True,
                        help="Gold BIO TSV — 2-col (token BIO), "
                             "3-col (token BIO cogroo_type), or "
                             "4-col (token BIO cogroo_type taxonomy)")
    parser.add_argument("--output", default="fn_sample_300.csv")
    parser.add_argument("--n", type=int, default=300,
                        help="Number of false negatives to sample (default: 300)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load predictions
    with open(args.predictions, encoding="utf-8") as f:
        data = json.load(f)
    predictions = data["predictions"]
    print(f"Loaded {len(predictions)} sentences from {args.predictions}")

    # Parse gold + generate taxonomy on the fly
    print(f"Parsing gold file and generating taxonomy from {args.gold} ...")
    span_taxonomy, _ = parse_gold_file(args.gold)
    print(f"  {len(span_taxonomy)} gold spans categorised")

    # Extract false negatives
    false_negatives = extract_false_negatives(predictions, span_taxonomy)
    print(f"  {len(false_negatives)} false negative spans found")

    # Stratified sample
    sampled = stratified_sample(false_negatives, args.n, seed=args.seed)

    # Report + export
    print_summary(false_negatives, sampled)
    export_csv(sampled, args.output)


if __name__ == "__main__":
    main()