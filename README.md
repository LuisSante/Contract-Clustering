# Contract-Clustering

Local discovery of themes in Portuguese credit-card contracts for a contract-focused adaptation of LegalXplain. This is a research pipeline and a proposed codebook, **not a trained legal classifier**.

It inventories native-text PDFs under `processed/`, deduplicates exact normalized texts, compares document clustering methods, discovers clause/window topics, and audits associations with issuer and extraction route. Input PDFs are read only. No document text is submitted to a remote inference service; model files are downloaded from Hugging Face on first semantic use.

## Install

Python 3.12+ and [uv](https://docs.astral.sh/uv/) are required for the locked environment:

```bash
uv sync --locked --extra semantic --extra dev
```

The semantic extra uses CPU PyTorch. For lexical discovery alone, omit `--extra semantic`. `uv.lock` pins dependencies; the default E5-small model is pinned to revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`. Every embedding run records the resolved revision, preprocessing signature, token window, pooling and library version. Seeds are fixed. Floating-point results can vary by platform. The model and revision can change via `--model` and `--revision`; the current implementation uses E5's `query: ` prefix and therefore expects an E5-compatible model, rather than arbitrary embeddings.

## Run

Set your own paths. Keep `WORK` **outside** the original contract folder and preferably outside this Git checkout:

```bash
INPUT='/path/to/Contratos'
WORK='/path/to/private-research-artifacts'
METADATA='/path/to/DataJudScrapper/external/dataset/dataset.jsonl'

uv run contract-clustering inventory \
  --input "$INPUT" --output "$WORK/corpus" --metadata "$METADATA"

uv run contract-clustering benchmark \
  --corpus "$WORK/corpus/documents.jsonl" --output "$WORK/benchmark"

uv run contract-clustering clauses \
  --corpus "$WORK/corpus/documents.jsonl" --output "$WORK/clauses"

uv run contract-clustering audit \
  --corpus "$WORK/corpus/documents.jsonl" \
  --taxonomy configs/taxonomy-v0.1.yaml --output "$WORK/taxonomy-signals.json"
```

`--metadata` is optional. Without it, case subjects are unavailable. Use `benchmark --lexical-only` for the lexical baseline without model downloads. Benchmark requires at least 20 distinct documents; clause discovery requires 40 distinct units. These commands assume a directory of lower-case `.pdf` files and native text; they do not perform OCR.

## What is produced

| Artifact | Meaning |
| --- | --- |
| `corpus/corpus_summary.json`, `inventory.json` | Physical counts, scope, failures, text hashes and duplicate audit |
| `corpus/documents.jsonl` | **Private:** text, local paths, source metadata, heuristic units and offsets |
| `benchmark/benchmark.csv` | All document experiments; coverage, silhouette, stability and confound audits |
| `benchmark/assignments.csv` | Local identifiers and assignments to every model; grouped evaluation IDs |
| `benchmark/clusters.html` | Standalone interactive projection and benchmark table; opens locally |
| `benchmark/clusters.json` | Representative snippets and terms for analyst review; **private** |
| `benchmark/near_duplicate_groups.json` | Template similarity groups and case/template union for future evaluation |
| `benchmark/clause_topics.json` | NMF topics over distinct clause/windows |
| `clauses/clause_clusters.json` | Semantic clause/window clusters, class TF-IDF terms and actual representative spans |
| `clauses/clause_benchmark.json` | Clause experiment metrics and embedding configuration |
| `taxonomy-signals.json` | Overlapping lexical mention counts for the proposed codebook, **not labeled examples** |
| `configs/taxonomy-v0.1.yaml` | Proposed 14 clause themes, definitions, exclusions, synthetic examples and additional axes |

## Interpretation

The extraction folders `indices`, `primera_pagina`, `buscar_texto`, and `contratos_genericos` describe how a PDF was found. They are **not semantic classes**. An entire contract normally contains several themes. A bank/template cluster is useful for sampling but should not become a legal theme merely because its silhouette is high.

The recommended product design is hierarchical, multilabel clause themes plus separate product/instrument facets and source-backed legal questions. Court case subjects remain contextual metadata. Contract text alone does not establish fraud, wrongful reporting, actual consumer intent, damages or contractual unfairness.

The taxonomy is provisional. Its lexical patterns measure textual signals and can overcount. Absence of a match is not absence of a provision, especially in redacted, partial or reconstructed text. It needs Portuguese legal annotation and a held-out evaluation before use as a classifier.

See [methodology and alternatives](docs/METHODOLOGY.md) and [integration proposal](docs/LEGALXPLAIN.md).

## Privacy and provenance

Do not commit PDFs, corpus text, snippets, case identifiers, API snapshots, model caches or generated assignments. `.gitignore` excludes common data/artifact paths and formats, but is not a privacy scanner. JSON/text outputs can contain sensitive material if placed elsewhere. An input named anonymized or processed has not thereby passed a new privacy/completeness audit.

Offsets identify text positions in the **processed PDF page**, not the original judicial case PDF. Reflowed PDFs need a separate geometric alignment before highlighting their original raster source. Exact-text deduplication retains local provenance; near-template groups are not deleted. The case/template union is prepared for a later grouped split; the current clustering study does not train or validate a supervised classifier.

## Checks

```bash
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
```

Tests cover input immutability and scope, exact deduplication, identifier/issuer masking, full long-document token coverage, processed-text offsets, density noise, evaluation grouping, overlapping taxonomy signals and an offline lexical pipeline smoke test.
