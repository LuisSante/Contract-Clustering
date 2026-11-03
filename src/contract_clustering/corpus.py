"""Read-only inventory of processed PDFs and local provenance; no uploads or OCR guessing."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pymupdf

from .text import canonical, clean, institutions, segments


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_corpus(root: Path, output: Path, metadata: Path | None = None):
    root, output = root.resolve(), output.resolve()
    if not root.is_dir():
        raise ValueError("input directory does not exist")
    if output.is_relative_to(root):
        raise ValueError("output must be outside the input corpus")
    output.mkdir(parents=True, exist_ok=True)
    case_metadata = {}
    if metadata:
        for line in metadata.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            case_metadata[row["process_number"]] = row
    documents, inventory, failures, duplicates = [], [], [], []
    seen = {}
    for path in sorted(root.rglob("*.pdf")):
        relative = path.relative_to(root)
        if path.name == "resumen_procesamiento.pdf":
            continue
        try:
            with pymupdf.open(path) as pdf:
                pages = [p.get_text(sort=True) for p in pdf]
        except (OSError, ValueError, RuntimeError) as error:
            failures.append({"path": str(relative), "error": type(error).__name__})
            continue
        text = "\n\f\n".join(pages)
        text_hash = hashlib.sha256(canonical(text).encode()).hexdigest()
        row = {
            "path": str(relative),
            "method": relative.parts[0],
            "processed": "processed" in relative.parts,
            "pages": len(pages),
            "chars": len(text),
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "text_sha256": text_hash,
        }
        inventory.append(row)
        if not row["processed"]:
            continue
        if len(text.strip()) < 300:
            failures.append({"path": str(relative), "error": "insufficient_native_text"})
            continue
        if text_hash in seen:
            duplicates.append({"path": str(relative), "canonical_doc_id": seen[text_hash]})
            continue
        doc_id = "doc_" + text_hash[:16]
        seen[text_hash] = doc_id
        match = re.search(r"\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}", path.name)
        source = case_metadata.get(match[0], {}) if match else {}
        detected = institutions(text)
        units = []
        for page, page_text in enumerate(pages, 1):
            for i, unit in enumerate(segments(page_text), 1):
                units.append(
                    {
                        **unit,
                        "unit_id": f"{doc_id}_p{page}_{i}",
                        "processed_page": page,
                        "doc_id": doc_id,
                    }
                )
        documents.append(
            {
                **row,
                "doc_id": doc_id,
                "text": text,
                "clean_text": clean(text),
                "issuer_preserved_text": clean(text, mask_institutions=False),
                "institutions": detected,
                "institution": detected[0]
                if len(detected) == 1
                else "mixed"
                if detected
                else "unknown",
                "subjects": source.get("datajud_subjects", []),
                "case_subject": source.get("subject"),
                "procedural_class": source.get("classe"),
                "metadata_matched": bool(source),
                "case_group": hashlib.sha256(match[0].encode()).hexdigest()[:20] if match else None,
                "case_year": int(match[0][11:15]) if match else None,
                "units": units,
            }
        )
    with (output / "documents.jsonl").open("w", encoding="utf-8") as stream:
        for row in documents:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "input": str(root),
        "physical_contract_pdfs": len(inventory),
        "processed_pdfs": sum(r["processed"] for r in inventory),
        "processed_by_method": dict(Counter(r["method"] for r in inventory if r["processed"])),
        "unique_processed_texts": len(documents),
        "exact_text_duplicates": len(duplicates),
        "pages_before_dedup": sum(r["pages"] for r in inventory if r["processed"]),
        "pages_after_dedup": sum(r["pages"] for r in documents),
        "native_text_outside_processed": sum(
            not r["processed"] and r["chars"] >= 300 for r in inventory
        ),
        "metadata_matched": sum(r["metadata_matched"] for r in documents),
        "institution_counts": dict(Counter(r["institution"] for r in documents)),
        "units": sum(len(r["units"]) for r in documents),
        "failures": failures,
        "duplicates": duplicates,
        "policy": "processed only; normalized exact-text dedup; original PDFs unchanged; quality is not a privacy certification",
    }
    write_json(output / "corpus_summary.json", summary)
    write_json(output / "inventory.json", inventory)
    return documents, summary
