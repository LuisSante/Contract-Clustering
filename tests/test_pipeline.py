import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from contract_clustering.audit import audit_taxonomy
from contract_clustering.benchmark import leakage_groups, metrics, run_benchmark, token_windows
from contract_clustering.corpus import build_corpus
from contract_clustering.text import clean, institutions, segments


def test_long_contract_windows_keep_final_tokens_and_overlap():
    tokens = list(range(3121))
    parts = list(token_windows(tokens))
    assert set().union(*(set(p) for p in parts)) == set(tokens)
    assert parts[-1][-1] == 3120
    assert parts[0][-32:] == parts[1][:32]
    assert all(len(p) <= 256 for p in parts)
    with pytest.raises(ValueError):
        list(token_windows(tokens, window=32, overlap=32))


def test_masking_removes_aliases_and_ids_but_preserves_legal_content():
    value = "BANCO PAN e Cetelem CPF 123.456.789-00, processo 1234567-89.2024.8.05.0001; juros de 4% e saldo da fatura. Panorama geral."
    result = clean(value)
    assert "PAN" not in result and "Cetelem" not in result and "1234567" not in result
    assert "juros" in result and "fatura" in result and "percentual" in result
    assert "Panorama" in result
    assert institutions(value) == ["pan", "cetelem"]
    assert "PAN" in clean(value, mask_institutions=False)


def test_offsets_are_in_original_processed_page_text():
    page = "1. O titular autoriza o pagamento da fatura por meio de desconto na folha de pagamento.\n2. O banco informa os juros e as tarifas incidentes sobre a operação de crédito contratada."
    spans = segments(page, max_words=40, overlap=3)
    assert spans
    for span in spans:
        assert clean(page[span["start"] : span["end"]]) == span["text"]


def test_inventory_scope_dedup_and_input_immutability(tmp_path, monkeypatch):
    root = tmp_path / "input"
    files = [
        root / "indices/processed/a.pdf",
        root / "buscar_texto/processed/b.pdf",
        root / "indices/original.pdf",
        root / "resumen_procesamiento.pdf",
    ]
    for file in files:
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"input stays unchanged " + file.name.encode())
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    text = "O titular declara conhecer as condições de pagamento da fatura e dos juros. " * 12

    class FakePDF:
        def __enter__(self):
            return [SimpleNamespace(get_text=lambda **kwargs: text)]

        def __exit__(self, *args):
            return False

    monkeypatch.setattr("contract_clustering.corpus.pymupdf.open", lambda p: FakePDF())
    records, summary = build_corpus(root, tmp_path / "artifacts")
    assert summary["physical_contract_pdfs"] == 3
    assert summary["processed_pdfs"] == 2
    assert summary["unique_processed_texts"] == 1
    assert summary["exact_text_duplicates"] == 1
    assert summary["native_text_outside_processed"] == 1
    assert len(records) == 1
    assert hashes == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    with pytest.raises(ValueError, match="outside"):
        build_corpus(root, root / "result")


def test_noise_is_counted_as_abstention_not_a_class():
    vectors = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9], [0.7, 0.7]])
    stats = metrics([0, 0, 1, 1, -1], vectors, ["a"] * 5, ["indices"] * 5)
    assert stats["clusters"] == 2 and stats["noise"] == 1
    assert stats["coverage"] == 0.8
    assert stats["silhouette_common_space"] > 0.8


def test_case_and_template_union_prevents_shared_case_split():
    result = leakage_groups([0, 0, 1, 2, 3], ["a", "b", "b", None, None])
    assert result[0] == result[1] == result[2]
    assert result[3] != result[4]
    assert result[3] != result[0]


def test_taxonomy_signals_are_overlapping_not_legal_verdicts(tmp_path):
    taxonomy = Path(__file__).resolve().parents[1] / "configs/taxonomy-v0.1.yaml"
    config = yaml.safe_load(taxonomy.read_text())
    ids = [t["id"] for t in config["clause_themes"]]
    assert len(ids) == len(set(ids)) == 14
    records = [
        {
            "clean_text": "Há seguro opcional e juros na fatura.",
            "institution": "unknown",
            "subjects": [],
            "procedural_class": None,
        }
    ]
    result = audit_taxonomy(records, taxonomy, tmp_path / "signals.json")
    supported = [t["id"] for t in result["themes"] if t["documents_with_signal"]]
    assert {"SEGUROS_BENEFICIOS", "FATURA_AMORTIZACAO", "JUROS_CET_TARIFAS"} <= set(supported)
    assert "not prevalence" in result["warning"]


def test_lexical_smoke_creates_auditable_outputs_without_models(tmp_path):
    themes = [
        "juros encargos tarifa fatura saldo amortizacao credito rotativo",
        "dados pessoais compartilhamento cadastro sigilo informacoes privacidade",
        "assinatura testemunhas adesao consentimento contrato biometria aceite",
    ]
    records = []
    for i in range(24):
        text = (themes[i % 3] + f" variante{i % 8} clausula{i % 6} ") * (2 + i % 4)
        records.append(
            {
                "doc_id": f"synthetic{i}",
                "clean_text": text,
                "issuer_preserved_text": text + " banco ficticio",
                "institution": f"issuer{i % 2}",
                "method": "indices",
                "pages": 1,
                "case_group": f"case{i // 2}",
            }
        )
    results, _, _ = run_benchmark(records, tmp_path, semantic=False)
    assert results
    assert (tmp_path / "clusters.html").is_file()
    summary = json.loads((tmp_path / "run_summary.json").read_text())
    assert summary["gold_labels"] is False
    assignments = (tmp_path / "assignments.csv").read_text()
    assert "evaluation_group" in assignments
    assert not (tmp_path / "embeddings.npy").exists()
