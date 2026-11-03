"""Descriptive lexical signals for a proposed taxonomy, explicitly not gold labels."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

import yaml

from .corpus import write_json


def fold(text):
    return "".join(
        c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c)
    )


def audit_taxonomy(records, taxonomy, output):
    codebook = yaml.safe_load(taxonomy.read_text(encoding="utf-8"))
    rows = []
    for theme in codebook["clause_themes"]:
        pattern = re.compile("|".join(f"(?:{p})" for p in theme["patterns"]))
        matching = [r for r in records if pattern.search(fold(r["clean_text"]))]
        rows.append(
            {
                "id": theme["id"],
                "name": theme["name"],
                "documents_with_signal": len(matching),
                "fraction": len(matching) / len(records) if records else 0,
                "institutions": dict(Counter(r["institution"] for r in matching)),
            }
        )
    result = {
        "documents": len(records),
        "taxonomy": codebook["version"],
        "themes": rows,
        "subjects": dict(Counter(s for r in records for s in set(r["subjects"]))),
        "procedural_classes": dict(Counter(r["procedural_class"] for r in records)),
        "case_years": dict(Counter(r.get("case_year") for r in records)),
        "case_groups": len({r.get("case_group") for r in records if r.get("case_group")}),
        "warning": "Lexical mention signals only: overlapping, unvalidated, not prevalence of legal violations or annotated classes. Case year is not contract date.",
    }
    write_json(output, result)
    return result
