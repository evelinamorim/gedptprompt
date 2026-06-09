"""
cogroo_taxonomy.py
==================
Maps CoGrOO rule identifiers and short messages to a unified 10-category
error taxonomy for Brazilian Portuguese GED.

Taxonomy (validated by Adrianna, PhD Linguistics):
  ortografia_lexical    – spelling, word-choice confusions (mal/mau, etc.)
  acentuacao            – missing or wrong accent marks (esta/está, e/é, etc.)
  concordancia_verbal   – subject-verb agreement
  concordancia_nominal  – noun-adjective / article-noun agreement
  crase_preposicao      – crase and preposition errors
  colocacao_pronominal  – clitic pronoun placement
  colocacao_verbal      – irregular verb form in tense/mood context
  classe_gramatical     – wrong grammatical class usage (e.g. inflecting an
                          invariable adverb: *meia cansada → meio cansada)
  sintatico_discursivo  – discourse connectors, punctuation, redundancy,
                          cohesion devices
  outros                – residual / not covered above

Change log:
  v2 (2026-06) – Expanded from 8 to 10 categories following validation by
                 Adrianna (PhD Linguistics):
                 · ortografia_acento renamed/merged into acentuacao
                 · classe_gramatical added (was mapped to outros)
                 · colocacao_verbal added (was mapped to concordancia_verbal)
                 · 19 rule-ID overrides applied from validated CSV
                 · classify_unknown_span updated with accent minimal-pair
                   detection and classe_gramatical heuristic
"""

from __future__ import annotations
import re
from pathlib import Path

# ------------------------------------------------------------------ #
# 1.  Rule-ID overrides  (Adrianna's validated corrections)
# ------------------------------------------------------------------ #
# These take highest priority — checked before short-message lookup.

RULE_ID_OVERRIDES: dict[str, str] = {
    "xml:38":  "classe_gramatical",    # "meio" as adverb — invariable
    "xml:39":  "classe_gramatical",    # "meio" as adverb — invariable
    "xml:42":  "concordancia_verbal",  # "fazer" impersonal — time
    "xml:43":  "concordancia_verbal",  # "fazer" impersonal — time
    "xml:46":  "classe_gramatical",    # "haver" impersonal — word class
    "xml:47":  "classe_gramatical",    # "haver" impersonal — word class
    "xml:48":  "colocacao_pronominal", # "entre eu e ele" → pronoun case
    "xml:49":  "colocacao_pronominal", # "entre eu e ele" → pronoun case
    "xml:50":  "colocacao_pronominal", # pronoun case (haver context)
    "xml:51":  "ortografia_lexical",   # mal/mau confusion
    "xml:59":  "colocacao_pronominal", # "preferir" + pronoun redundancy
    "xml:75":  "colocacao_pronominal", # subjunctive + clitic placement
    "xml:76":  "colocacao_pronominal", # subjunctive + clitic placement
    "xml:77":  "colocacao_verbal",     # "dizer" → "disser" (fut. subj.)
    "xml:92":  "concordancia_nominal", # "meio-dia e meia" agreement
    "xml:106": "sintatico_discursivo", # "à medida em que" → expression
    "xml:109": "crase_preposicao",     # "assistir" + crase
    "xml:119": "concordancia_verbal",  # "haver" impersonal (existir)
    "xml:122": "sintatico_discursivo", # "há dois anos atrás" — redundancy
}

# ------------------------------------------------------------------ #
# 2.  Short-message → taxonomy  (primary lookup)
# ------------------------------------------------------------------ #

COGROO_TO_TAXONOMY: dict[str, str] = {
    # ── Concordância verbal ──────────────────────────────────────────
    "Verificou-se erro de concordância entre o sujeito e o verbo.":
        "concordancia_verbal",
    "O adjetivo na função de predicativo concorda com o verbo.":
        "concordancia_verbal",
    "O sujeito concorda em número com o predicado.":
        "concordancia_verbal",

    # ── Concordância nominal ─────────────────────────────────────────
    "O adjetivo concorda com o substantivo a que se refere.":
        "concordancia_nominal",
    "O adjetivo na função de predicativo concorda com o sujeito.":
        "concordancia_nominal",
    "O predicativo concorda com o sujeito.":
        "concordancia_nominal",
    "Os determinantes concordam com o substantivo a que se referem.":
        "concordancia_nominal",
    "Os artigos concordam com o substantivo a que se referem.":
        "concordancia_nominal",
    "Os numerais concordam com o substantivo a que se referem.":
        "concordancia_nominal",

    # ── Crase / preposição ───────────────────────────────────────────
    "Não ocorre crase antes de palavras masculinas.":
        "crase_preposicao",
    "Não acontece crase antes de verbo.":
        "crase_preposicao",
    "Não há crase neste caso, somente no plural (\"às\").":
        "crase_preposicao",
    "Alguns nomes regem a preposição \"a\", logo há crase aqui.":
        "crase_preposicao",
    "\"Devido\" rege a preposição \"a\", logo há crase aqui.":
        "crase_preposicao",
    "\"(Des)obedecer\" constrói-se com prep. \"a\". Há crase com compl. feminino.":
        "crase_preposicao",
    "\"Em relação\" rege a preposição \"a\", logo há crase aqui.":
        "crase_preposicao",
    "\"Com relação\" rege a preposição \"a\", logo há crase aqui.":
        "crase_preposicao",
    "Ocorre crase em expressões indicativas de horas.":
        "crase_preposicao",
    "Ocorre crase em expressões como esta.":
        "crase_preposicao",
    "Não há crase porque alguns pronomes pessoais não admitem artigo.":
        "crase_preposicao",
    "Pronomes de tratamento não admitem artigo, logo não há crase aqui.":
        "crase_preposicao",
    "\"Ir\" constrói-se com prep. \"a\". Há crase com compl. feminino.":
        "crase_preposicao",
    "\"Aderir\" constrói-se com preposição \"a\". Se o complemento for feminino, teremos crase.":
        "crase_preposicao",
    "\"Pertencer\" constrói-se com a preposição \"a\". Se o complemento for feminino, teremos crase.":
        "crase_preposicao",
    "\"Candidatar-se\" constrói-se com prep. \"a\". Há crase com compl. feminino.":
        "crase_preposicao",
    "\"Reagir\" constrói-se com a preposição \"a\". Há crase com compl. feminino.":
        "crase_preposicao",
    "\"Equivalente\" rege a preposição \"a\". Com compl. feminino, há crase.":
        "crase_preposicao",
    "\"Equivaler\" constrói-se com prep. \"a\". Há crase com compl. feminino.":
        "crase_preposicao",
    "\"Ir\" rege preposição \"a\". Se compl. feminino, há crase.":
        "crase_preposicao",
    "Não há crase pois o \"a\" é apenas preposição.":
        "crase_preposicao",

    # ── Colocação pronominal ─────────────────────────────────────────
    "Pron. relativos e conj. subordinativas atraem o pronome para antes do verbo.":
        "colocacao_pronominal",
    "Palavras negativas atraem o pronome átono para antes do verbo.":
        "colocacao_pronominal",
    "Certos advérbios atraem o pronome para antes do verbo.":
        "colocacao_pronominal",
    "Alguns pronomes indefinidos atraem o pronome para antes do verbo.":
        "colocacao_pronominal",
    "\"só, ou, ora e quer\" atraem o pronome para antes do verbo.":
        "colocacao_pronominal",
    "\"Eu\" não deve ser preposicionado. Use \"a mim\".":
        "colocacao_pronominal",
    "O pronome \"eu\" não pode ser regido de preposição.":
        "colocacao_pronominal",
    "Use \"eu\" ao invés de \"mim\" como sujeito de verbo no infinitivo.":
        "colocacao_pronominal",

    # ── Regência ─────────────────────────────────────────────────────
    "Regência verbal.":
        "regencia",
    "Os verbos \"evitar\" e \"usufruir\" não regem preposição \"de\".":
        "regencia",
    "\"(Des)obedecer\" constrói-se com preposição \"a\".":
        "regencia",
    "Atenção para a regência do verbo \"habituar-se\": use \"habituar-se a\" em lugar de \"habituar-se com\".":
        "regencia",
    "\"Valorização\" rege preposição \"de\" e não preposição \"a\".":
        "regencia",
    "Atenção para a regência de alguns verbos.":
        "regencia",
    "Não use a preposição \"com\" na regência do verbo namorar.":
        "regencia",
    "O verbo \"arrasar\" não rege preposição \"com\".":
        "regencia",

    # ── Ortografia lexical ───────────────────────────────────────────
    "O contrário de \"bem\" é \"mal\", e o de \"bom\" é \"mau\".":
        "ortografia_lexical",

    # ── Sintático / discursivo ───────────────────────────────────────
    "Repetição de palavras.":
        "sintatico_discursivo",
    "Deve haver vírgula antes de \"no entanto\".":
        "sintatico_discursivo",
    "A expressão \"ou seja\" deve ser isolada por vírgulas.":
        "sintatico_discursivo",
    "Verifique o excesso de verbos em sequência.":
        "sintatico_discursivo",
    "A expressão correta é \"à medida que\".":
        "sintatico_discursivo",
    "\"Conviver junto\" ou \"conviver juntos\" são expressões redundantes.":
        "sintatico_discursivo",
    "\"Atrás\" é redundante, devido ao uso do verbo \"haver\".":
        "sintatico_discursivo",
    "Suprima o \"mais\". O sentido de preferir é \"querer mais\".":
        "sintatico_discursivo",
    "A expressão \"em anexo\" é invariável.":
        "sintatico_discursivo",

    # ── Classe gramatical ────────────────────────────────────────────
    "A palavra \"meio\" usada no sentido de \"um pouco\" (advérbio) é invariável.":
        "classe_gramatical",
    "\"Meia\" concorda com o substantivo \"hora\", subentendido.":
        "classe_gramatical",
    "Opte pelo verbo \"haver\", no singular, para indicar tempo decorrido.":
        "classe_gramatical",
    "As formas do verbo \"haver\" ficam no singular quando indicam tempo decorrido.":
        "classe_gramatical",
    "\"Haver\" no sentido de existir é usado na 3a. pessoa do singular.":
        "classe_gramatical",
    "\"Haver\" (existir), precedido de verbo aux., é usado na 3a. do singular.":
        "classe_gramatical",
    "\"Fazer\", quando indica tempo, deve permanecer no singular.":
        "classe_gramatical",
}

# ------------------------------------------------------------------ #
# 3.  Regex patterns for message-level classification
# ------------------------------------------------------------------ #

# "Possível confusão entre X e X." → acentuacao (e.g. está/esta, é/e)
_CONFUSION_RE = re.compile(r"Possível confusão entre .+ e .+\.", re.IGNORECASE)

# ------------------------------------------------------------------ #
# 4.  Public API: CoGrOO message → taxonomy
# ------------------------------------------------------------------ #

def cogroo_msg_to_taxonomy(msg: str, rule_id: str | None = None) -> str:
    """
    Map a CoGrOO short_msg (and optional rule_id) to a taxonomy category.

    Priority:
      1. RULE_ID_OVERRIDES (Adrianna's explicit corrections)
      2. COGROO_TO_TAXONOMY (short-message lookup)
      3. _CONFUSION_RE     (accent confusion pattern → acentuacao)
      4. Sentinel values   (-, UNKNOWN, NO_MATCH) passed through as-is
      5. Fallback          → "UNKNOWN"
    """
    if rule_id and rule_id in RULE_ID_OVERRIDES:
        return RULE_ID_OVERRIDES[rule_id]
    if msg in ("-", "UNKNOWN", "NO_MATCH"):
        return msg
    if msg in COGROO_TO_TAXONOMY:
        return COGROO_TO_TAXONOMY[msg]
    if _CONFUSION_RE.match(msg):
        return "acentuacao"
    return "UNKNOWN"

# ------------------------------------------------------------------ #
# 5.  Heuristic taxonomy for UNKNOWN spans (no CoGrOO label)
# ------------------------------------------------------------------ #

# Accentuation minimal pairs: (unaccented, accented) – both directions.
# Only include pairs where the two forms are genuinely different words
# (one with, one without accent), making the error orthographic in nature.
# Do NOT include demonstrative pronouns (aquele/àquele etc.) — those differ
# only by crase marker and belong to crase_preposicao, not acentuacao.
_ACCENT_PAIRS: frozenset[frozenset[str]] = frozenset(
    frozenset(pair) for pair in [
        ("esta",  "está"),    # demonstrative vs verb estar
        ("e",     "é"),       # conjunction vs verb ser
        ("a",     "á"),       # article/prep vs interjection
        ("de",    "dê"),      # preposition vs verb dar (subj.)
        ("por",   "pôr"),     # preposition vs verb pôr
        ("pelo",  "pêlo"),    # contraction vs noun (hair)
        ("para",  "pará"),    # preposition vs state name
        ("polo",  "pôlo"),    # contraction vs noun
        ("vem",   "vêm"),     # singular vs plural of vir
        ("tem",   "têm"),     # singular vs plural of ter
        ("o",     "ô"),       # article vs interjection
        ("pode",  "pôde"),    # present vs preterite of poder
        ("mais",  "más"),     # adverb vs adjective (bad, fem. pl.)
    ]
)

# Crase tokens: single tokens that are definitively crase/preposition errors,
# including demonstrative pronouns with a crase marker.
# Adrianna's note: àquele/àqueles are exceptions to the "no crase before
# masculine words" rule — crase IS valid before demonstrative pronouns,
# so CoGrOO's rule fires incorrectly on them. We ensure they classify as
# crase_preposicao (not acentuacao) in the heuristic.
_CRASE_TOKENS: frozenset[str] = frozenset([
    "a", "as", "ao", "aos", "à", "às",
    "àquele", "àquela", "àqueles", "àquelas", "àquilo",
])

# Tokens that are invariable adverbs but commonly inflected (classe_gramatical)
_INVARIABLE_ADVERBS: frozenset[str] = frozenset([
    "meio", "meia", "meios", "meias",   # "meia cansada" error
    "mesmo", "mesma",                    # when used as adverb
    "bastante", "bastantes",
    "menos", "mais",
])

_VERBAL_RE = re.compile(
    r"\b(foi|foram|era|eram|é|são|está|estão|tem|têm|vem|vêm|"
    r"ser|estar|ter|haver|faz|fazem|vai|vão|deve|devem|pode|podem|"
    r"haviam|haveriam|faziam|faríamos)\b",
    re.IGNORECASE,
)
_NOMINAL_RE = re.compile(
    r"\b(os|as|uns|umas|maus|más|bons|boas|todos|todas|"
    r"muitos|muitas|poucos|poucas|outros|outras)\b",
    re.IGNORECASE,
)
_CRASE_RE = re.compile(r"^(a|as|ao|aos|à|às|àquele|àquela|àqueles|àquelas|àquilo)$", re.IGNORECASE)
_HAS_ACCENT_RE = re.compile(r"[àáâãéêíóôõúç]")
_PRONOUN_RE = re.compile(
    r"\b(se|me|te|lhe|nos|vos|lhes|o|a|os|as|mim|ti|si|eu)\b",
    re.IGNORECASE,
)
_SUBJUNCTIVE_IRREGULAR_RE = re.compile(
    r"\b(dizer|fazer|trazer|querer|poder|vir|ver|vir|ser|ir|"
    r"dizer|pôr|saber)\b",
    re.IGNORECASE,
)


def classify_unknown_span(span_tokens: list[str]) -> str:
    """
    Heuristic classification of UNKNOWN error spans (no CoGrOO label).

    Checks in order:
      1. Any token is an invariable adverb  → classe_gramatical
      2. Single token is crase candidate    → crase_preposicao  (takes priority
                                               over accent check: à/a are valid
                                               words used in wrong context)
      3. Single token in accent pair        → acentuacao
      4. Single token has accent            → acentuacao
      5. Single token (no accent)           → ortografia_lexical
      6. Multi-token: verbal agreement cue  → concordancia_verbal
      7. Multi-token: nominal agreement cue → concordancia_nominal
      8. Multi-token: pronoun (≤3 tokens)   → colocacao_pronominal
      9. Multi-token: irregular verb form   → colocacao_verbal
     10. Long span (≥3 tokens)              → sintatico_discursivo
     11. Fallback                           → ortografia_lexical
    """
    text = " ".join(span_tokens)
    lower_tokens = [t.lower() for t in span_tokens]

    # ── 1. Invariable adverb inflected as adjective ─────────────────
    if any(t in _INVARIABLE_ADVERBS for t in lower_tokens):
        return "classe_gramatical"

    # ── 2–5. Single-token cases ──────────────────────────────────────
    if len(span_tokens) == 1:
        tok = lower_tokens[0]
        # Crase candidates (à, às, a, as, ao, aos) take priority —
        # these are syntactically valid words used in wrong context
        if _CRASE_RE.match(tok):
            return "crase_preposicao"
        # Accent minimal-pair check (e.g. esta/está, e/é, por/pôr)
        for pair in _ACCENT_PAIRS:
            if tok in pair:
                return "acentuacao"
        # Single accented token not in a pair → still likely accent error
        if _HAS_ACCENT_RE.search(tok):
            return "acentuacao"
        return "ortografia_lexical"

    # ── 6. Multi-token: verbal agreement ────────────────────────────
    if _VERBAL_RE.search(text):
        return "concordancia_verbal"

    # ── 7. Multi-token: nominal agreement ───────────────────────────
    if _NOMINAL_RE.search(text):
        return "concordancia_nominal"

    # ── 8. Multi-token: pronoun placement (short span) ──────────────
    if _PRONOUN_RE.search(text) and len(span_tokens) <= 3:
        return "colocacao_pronominal"

    # ── 9. Irregular verb form in tense/mood ────────────────────────
    if _SUBJUNCTIVE_IRREGULAR_RE.search(text):
        return "colocacao_verbal"

    # ── 10. Longer span → likely discourse / syntactic ──────────────
    if len(span_tokens) >= 3:
        return "sintatico_discursivo"

    # ── 11. Fallback ────────────────────────────────────────────────
    return "ortografia_lexical"


# ------------------------------------------------------------------ #
# 6.  Gold file parsing → per-sentence span taxonomy map
# ------------------------------------------------------------------ #

_SENTENCE_HEADER_RE = re.compile(r"^Sentença", re.IGNORECASE)

def build_span_taxonomy_from_gold(
    gold_path: str | Path,
) -> dict[int, dict[tuple[int, int], str]]:
    """
    Parse a typed gold TSV (4-column: token, BIO, rule_id, short_msg) and
    return a nested dict:
        { sentence_id: { (start_idx, end_idx): taxonomy_label } }

    Falls back to classify_unknown_span for spans with no CoGrOO label.
    """
    result: dict[int, dict[tuple[int, int], str]] = {}
    sentence_id = 0
    current_tokens: list[str] = []
    current_labels: list[str] = []
    current_rule_ids: list[str] = []
    current_msgs: list[str] = []

    def flush_sentence():
        if not current_tokens:
            return
        spans = _extract_spans(
            current_tokens, current_labels, current_rule_ids, current_msgs
        )
        result[sentence_id] = spans

    with open(gold_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if _SENTENCE_HEADER_RE.match(line):
                flush_sentence()
                sentence_id += 1
                current_tokens.clear()
                current_labels.clear()
                current_rule_ids.clear()
                current_msgs.clear()
                continue
            if not line.strip():
                continue
            parts = line.split("\t")
            token = parts[0] if len(parts) > 0 else ""
            bio   = parts[1] if len(parts) > 1 else "O"
            rid   = parts[2] if len(parts) > 2 else ""
            msg   = parts[3] if len(parts) > 3 else ""
            current_tokens.append(token)
            current_labels.append(bio)
            current_rule_ids.append(rid)
            current_msgs.append(msg)

    flush_sentence()
    return result


def _extract_spans(
    tokens: list[str],
    labels: list[str],
    rule_ids: list[str],
    msgs: list[str],
) -> dict[tuple[int, int], str]:
    """Extract error spans and assign taxonomy labels."""
    spans: dict[tuple[int, int], str] = {}
    i = 0
    while i < len(labels):
        if labels[i].startswith("B-"):
            start = i
            span_tokens = [tokens[i]]
            span_rids   = [rule_ids[i]]
            span_msgs   = [msgs[i]]
            j = i + 1
            while j < len(labels) and labels[j].startswith("I-"):
                span_tokens.append(tokens[j])
                span_rids.append(rule_ids[j])
                span_msgs.append(msgs[j])
                j += 1
            # Pick the first non-empty rule_id / msg in the span
            rid = next((r for r in span_rids if r.strip()), "")
            msg = next((m for m in span_msgs if m.strip()), "")
            label = cogroo_msg_to_taxonomy(msg, rule_id=rid)
            if label == "UNKNOWN":
                label = classify_unknown_span(span_tokens)
            spans[(start, j - 1)] = label
            i = j
        else:
            i += 1
    return spans


# ------------------------------------------------------------------ #
# 7.  Taxonomy constants (for use in other modules)
# ------------------------------------------------------------------ #

TAXONOMY_CATEGORIES: tuple[str, ...] = (
    "ortografia_lexical",
    "acentuacao",
    "concordancia_verbal",
    "concordancia_nominal",
    "crase_preposicao",
    "colocacao_pronominal",
    "colocacao_verbal",
    "classe_gramatical",
    "sintatico_discursivo",
    "outros",
)

# ------------------------------------------------------------------ #
# 8.  File processing
# ------------------------------------------------------------------ #

_SENTENCE_RE = re.compile(r"^Sentença", re.IGNORECASE)


def add_taxonomy_column(input_path: str | Path, output_path: str | Path) -> None:
    """
    Read a 3-column BIO TSV (token  BIO_label  cogroo_type) and write a
    4-column version with the taxonomy category added as the fourth column.

    The cogroo_type column may contain either a CoGrOO short_msg string or
    a rule_id of the form 'xml:N'.  Both are handled via cogroo_msg_to_taxonomy.
    When cogroo_type is 'UNKNOWN', the heuristic classify_unknown_span is used.
    """
    from collections import Counter, defaultdict

    # First pass: read all sentences
    sentences: list[tuple[int, list[str], list[str], list[str]]] = []
    sentence_id = 0
    current_tokens: list[str] = []
    current_labels: list[str] = []
    current_types:  list[str] = []

    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if _SENTENCE_RE.match(line):
                if current_tokens:
                    sentences.append(
                        (sentence_id, current_tokens[:], current_labels[:], current_types[:])
                    )
                sentence_id += 1
                current_tokens, current_labels, current_types = [], [], []
            elif not line.strip():
                if current_tokens:
                    sentences.append(
                        (sentence_id, current_tokens[:], current_labels[:], current_types[:])
                    )
                    current_tokens, current_labels, current_types = [], [], []
            else:
                parts = line.split("\t")
                current_tokens.append(parts[0])
                current_labels.append(parts[1].strip() if len(parts) > 1 else "O")
                current_types.append(parts[2].strip() if len(parts) > 2 else "-")
    if current_tokens:
        sentences.append((sentence_id, current_tokens, current_labels, current_types))

    # Compute taxonomy per token, span by span
    sent_taxonomy: dict[int, list[str]] = {}
    for sid, tokens, labels, types in sentences:
        taxonomy = ["-"] * len(tokens)
        i = 0
        while i < len(labels):
            if labels[i] == "B-WRONG":
                j = i + 1
                while j < len(labels) and labels[j] == "I-WRONG":
                    j += 1
                span_type = types[i]
                if span_type == "UNKNOWN":
                    cat = classify_unknown_span(tokens[i:j])
                elif span_type == "NO_MATCH":
                    cat = "NO_MATCH"
                elif span_type in ("-", ""):
                    cat = "-"
                else:
                    # span_type is either:
                    #   "rule_id|short_msg"  (new format from cogroo_annotate.py)
                    #   "short_msg"          (legacy format, no pipe)
                    if "|" in span_type:
                        rule_id, msg = span_type.split("|", 1)
                    else:
                        rule_id, msg = "", span_type
                    cat = cogroo_msg_to_taxonomy(msg, rule_id=rule_id or None)
                    if cat == "UNKNOWN":
                        cat = classify_unknown_span(tokens[i:j])
                for k in range(i, j):
                    taxonomy[k] = cat
                i = j
            elif labels[i] == "O":
                if types[i] == "NO_MATCH":
                    taxonomy[i] = "NO_MATCH"
                i += 1
            else:
                i += 1
        sent_taxonomy[sid] = taxonomy

    # Second pass: write output with fourth column
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
                if len(parts) >= 2:
                    tax = sent_taxonomy.get(sentence_id, [])
                    cat = tax[tok_idx] if tok_idx < len(tax) else "-"
                    fout.write("\t".join(parts) + f"\t{cat}\n")
                    tok_idx += 1
                else:
                    fout.write(line + "\n")

    print(f"Written: {output_path}")


# ------------------------------------------------------------------ #
# 9.  Statistics
# ------------------------------------------------------------------ #

def compute_statistics(typed_path: str | Path) -> dict:
    """
    Compute error-type statistics from a 4-column BIO TSV file.
    Returns a dict with span-level and token-level counts.
    """
    from collections import Counter

    span_taxonomy_counts: Counter = Counter()
    token_taxonomy_counts: Counter = Counter()
    cogroo_raw_counts: Counter = Counter()
    no_match_spans = 0
    total_gold_spans = 0
    total_tokens = 0

    current_tokens: list[str] = []
    current_labels: list[str] = []
    current_types:  list[str] = []
    current_taxonomy: list[str] = []

    def flush() -> None:
        nonlocal no_match_spans, total_gold_spans, total_tokens

        total_tokens += len(current_tokens)

        for lbl, cogroo, tax in zip(current_labels, current_types, current_taxonomy):
            token_taxonomy_counts[tax] += 1
            if cogroo not in ("-", "UNKNOWN", "NO_MATCH"):
                cogroo_raw_counts[cogroo] += 1

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

    return {
        "total_tokens":          total_tokens,
        "total_gold_spans":      total_gold_spans,
        "cogroo_matched_spans":  cogroo_matched,
        "unknown_spans":         span_taxonomy_counts.get("UNKNOWN", 0),
        "no_match_spans":        no_match_spans,
        "span_taxonomy":         dict(span_taxonomy_counts),
        "token_taxonomy":        dict(token_taxonomy_counts),
        "cogroo_raw":            dict(cogroo_raw_counts),
    }


def print_statistics(stats: dict, split_name: str) -> None:
    """Print a formatted statistics report for one split."""
    total_spans = stats["total_gold_spans"]
    pct = lambda n: f"({100 * n / max(total_spans, 1):.1f}%)"

    print(f"\n{'='*60}")
    print(f"  Split: {split_name}")
    print(f"{'='*60}")
    print(f"  Total tokens          : {stats['total_tokens']:>8}")
    print(f"  Total gold spans      : {total_spans:>8}")
    print(f"  CoGrOO matched spans  : {stats['cogroo_matched_spans']:>8}  "
          f"{pct(stats['cogroo_matched_spans'])}")
    print(f"  UNKNOWN spans         : {stats['unknown_spans']:>8}  "
          f"{pct(stats['unknown_spans'])}")
    print(f"  NO_MATCH spans        : {stats['no_match_spans']:>8}")

    print(f"\n  --- Span-level taxonomy distribution ---")
    total_tax = sum(stats["span_taxonomy"].values())
    for cat, count in sorted(stats["span_taxonomy"].items(), key=lambda x: -x[1]):
        pct_s = f"{100 * count / total_tax:.1f}%" if total_tax else "0.0%"
        print(f"    {cat:<35} {count:>6}  ({pct_s})")

    print(f"\n  --- CoGrOO raw categories (matched spans only) ---")
    for msg, count in sorted(stats["cogroo_raw"].items(), key=lambda x: -x[1]):
        print(f"    {msg:<60} {count:>5}")


# ------------------------------------------------------------------ #
# 10. __main__
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    splits = ["train", "val", "test"]
    all_stats: dict[str, dict] = {}

    for split in splits:
        input_path  = Path(f"data/{split}_bio_typed.tsv")
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
        all_cats = sorted(
            set(cat for s in all_stats.values() for cat in s["span_taxonomy"]),
            key=lambda c: (
                TAXONOMY_CATEGORIES.index(c) if c in TAXONOMY_CATEGORIES else 999
            ),
        )
        header = f"  {'Category':<35}" + "".join(f"{s:>10}" for s in all_stats)
        print(header)
        print(f"  {'-' * (35 + 10 * len(all_stats))}")
        for cat in all_cats:
            row = f"  {cat:<35}"
            for s_stats in all_stats.values():
                total = sum(s_stats["span_taxonomy"].values())
                count = s_stats["span_taxonomy"].get(cat, 0)
                row += f"  {100 * count / total:>6.1f}%" if total else "       —"
            print(row)