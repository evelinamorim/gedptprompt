"""
cogroo_rules_csv.py
Parses the CoGrOO rules.xml file and generates a CSV for linguist validation.

The XML structure per rule:
  <Rule id="N" active="true/false">
    <Type>...</Type>
    <Group>...</Group>
    <Message>...</Message>
    <ShortMessage>...</ShortMessage>
  </Rule>

Output CSV columns:
  rule_id           : CoGrOO rule ID (e.g. xml:1)
  active            : whether the rule is active in CoGrOO
  type              : CoGrOO rule type (e.g. Crase, Concordância)
  group             : CoGrOO rule group (sub-category)
  short_message     : short human-readable description (Portuguese)
  message           : full description (Portuguese)
  taxonomy_auto     : automatically assigned taxonomy category
  taxonomy_validated: empty column for linguist to fill
  notes             : empty column for linguist comments
  in_dataset        : whether this rule appeared in the dataset
  count_train/val/test/total : token occurrences per split

Usage:
    python cogroo_rules_csv.py --rules cogroo_rules.xml --output cogroo_rules_validation.csv
"""

from __future__ import annotations
import argparse
import csv
import re
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter
from pathlib import Path

TYPE_TO_TAXONOMY: dict[str, str] = {
    "Crase": "crase_preposicao",
    "Concordância": "concordancia_verbal",
    "Colocação": "colocacao_pronominal",
    "Regência": "regencia",
    "Uso do acento": "ortografia_acento",
    "Ortografia": "ortografia_lexical",
    "Repetição": "sintatico_discursivo",
    "Pontuação": "sintatico_discursivo",
    "Uso do hífen": "ortografia_lexical",
    "Uso do til": "ortografia_acento",
    "Estilo": "sintatico_discursivo",
    "Uso do porquê": "ortografia_lexical",
    "Uso do mal/mau": "ortografia_lexical",
}

GROUP_TAXONOMY_HINTS: dict[str, str] = {
    "sujeito": "concordancia_verbal",
    "verbo": "concordancia_verbal",
    "adjetivo": "concordancia_nominal",
    "artigo": "concordancia_nominal",
    "substantivo": "concordancia_nominal",
    "pronome": "colocacao_pronominal",
    "preposição": "crase_preposicao",
    "acento": "ortografia_acento",
    "ortograf": "ortografia_lexical",
    "vírgula": "sintatico_discursivo",
    "repetição": "sintatico_discursivo",
    "regência": "regencia",
}

_CONFUSION_RE = re.compile(r"Possível confusão entre .+ e .+\.", re.IGNORECASE)

SHORT_MSG_TO_TAXONOMY: dict[str, str] = {
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


def auto_taxonomy(rule_type: str, group: str, short_msg: str) -> str:
    if short_msg in SHORT_MSG_TO_TAXONOMY:
        return SHORT_MSG_TO_TAXONOMY[short_msg]
    if _CONFUSION_RE.match(short_msg):
        return "ortografia_acento"
    for type_key, tax in TYPE_TO_TAXONOMY.items():
        if type_key.lower() in rule_type.lower():
            if tax == "concordancia_verbal":
                group_lower = group.lower()
                if any(k in group_lower for k in ["adjetivo", "artigo", "substantivo", "determinante", "numeral"]):
                    return "concordancia_nominal"
            return tax
    group_lower = group.lower()
    for keyword, tax in GROUP_TAXONOMY_HINTS.items():
        if keyword in group_lower:
            return tax
    return "outros"


_SENTENCE_RE = re.compile(r"^Sentença", re.IGNORECASE)


def collect_dataset_counts(typed_files: dict[str, Path]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for split, path in typed_files.items():
        if not path.exists():
            print(f"  Skipping {split} — {path} not found")
            continue
        print(f"  Reading {path}...")
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if _SENTENCE_RE.match(line) or not line.strip():
                    continue
                parts = line.split("\t")
                if len(parts) >= 3:
                    cogroo_type = parts[2].strip()
                    if cogroo_type not in ("-", "UNKNOWN", "NO_MATCH"):
                        counts[cogroo_type][split] += 1
    return dict(counts)


def parse_rules_xml(xml_path: str | Path) -> list[dict]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    rules = []
    for rule_elem in root.findall("Rule"):
        rule_id = rule_elem.get("id", "")
        active = rule_elem.get("active", "true")
        rule_type = rule_elem.findtext("Type", "").strip()
        group = rule_elem.findtext("Group", "").strip()
        message = rule_elem.findtext("Message", "").strip()
        short_msg = rule_elem.findtext("ShortMessage", "").strip()
        rules.append({
            "rule_id": f"xml:{rule_id}",
            "active": active,
            "type": rule_type,
            "group": group,
            "short_message": short_msg,
            "message": message,
        })
    return rules


def generate_csv(xml_path: str | Path, output_path: str | Path) -> None:
    print(f"Parsing {xml_path}...")
    rules = parse_rules_xml(xml_path)
    print(f"  Found {len(rules)} rules ({sum(1 for r in rules if r['active'] == 'true')} active)")

    typed_files = {
        "train": Path("data/train_bio_typed.tsv"),
        "val":   Path("data/val_bio_typed.tsv"),
        "test":  Path("data/test_bio_typed.tsv"),
    }
    print("Collecting dataset occurrence counts...")
    dataset_counts = collect_dataset_counts(typed_files)

    rows = []
    for rule in rules:
        short_msg = rule["short_message"]
        taxonomy = auto_taxonomy(rule["type"], rule["group"], short_msg)
        counts = dataset_counts.get(short_msg, {})
        count_train = counts.get("train", 0)
        count_val   = counts.get("val", 0)
        count_test  = counts.get("test", 0)
        count_total = count_train + count_val + count_test
        rows.append({
            "rule_id":            rule["rule_id"],
            "active":             rule["active"],
            "type":               rule["type"],
            "group":              rule["group"],
            "short_message":      short_msg,
            "message":            rule["message"],
            "taxonomy_auto":      taxonomy,
            "taxonomy_validated": "",
            "notes":              "",
            "in_dataset":         "yes" if count_total > 0 else "no",
            "count_train":        count_train,
            "count_val":          count_val,
            "count_test":         count_test,
            "count_total":        count_total,
        })

    rows.sort(key=lambda r: (r["active"] != "true", -r["count_total"], r["rule_id"]))

    fieldnames = [
        "rule_id", "active", "type", "group",
        "short_message", "message",
        "taxonomy_auto", "taxonomy_validated", "notes",
        "in_dataset", "count_train", "count_val", "count_test", "count_total",
    ]

    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV written to: {output_path}")
    print(f"Total rules       : {len(rows)}")
    print(f"Active rules      : {sum(1 for r in rows if r['active'] == 'true')}")
    print(f"Rules in dataset  : {sum(1 for r in rows if r['in_dataset'] == 'yes')}")
    print(f"Rules not in dataset: {sum(1 for r in rows if r['in_dataset'] == 'no')}")

    tax_counts = Counter(r["taxonomy_auto"] for r in rows if r["active"] == "true")
    print(f"\nAuto taxonomy distribution (active rules):")
    for tax, count in sorted(tax_counts.items(), key=lambda x: -x[1]):
        in_ds = sum(1 for r in rows if r["taxonomy_auto"] == tax and r["in_dataset"] == "yes")
        print(f"  {tax:<35} {count:>4} rules  ({in_ds} in dataset)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse CoGrOO rules.xml and generate validation CSV")
    parser.add_argument("--rules", default="cogroo_rules.xml")
    parser.add_argument("--output", default="cogroo_rules_validation.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_csv(args.rules, args.output)