# Discovery methodology and alternatives

## Research question

Find candidate ways to organize Portuguese consumer credit-card contracts for an explainable reading interface. The task is to discover useful categories and evidence, not to predict a lawsuit outcome or prove an abusive practice. Methods and taxonomy designs are separate decisions.

## Units and preprocessing

1. Inventory all PDFs, excluding the processing summary. Analyze only PDFs in a `processed` path component and at least 300 native characters. Report excluded/native-text material rather than silently expanding scope.
2. Normalize Unicode and whitespace for exact text hashes. One analytical document per exact normalized text; retain duplicate paths locally. Distinct documents can still belong to one court case.
3. Remove judicial identifiers, long numeric strings, dates, URLs and known tribunal footers. Mask known issuer/brand aliases for primary views and keep a separate issuer-preserving control. This is feature control, not a new anonymization certification. Addresses, writing style and incomplete aliases can still reveal templates.
4. Keep the full legal body for document embeddings: overlapping 256-token windows, overlap 32, normalized E5 vectors and a length-weighted document mean followed by L2 normalization. No first-page or first-512-token document truncation. Overlap means boundary tokens receive additional weight; this is a simple discovery baseline, not learned document attention.
5. Segment processed page text using clause-like starts and 180-word windows with 25-word overlap. Discard units shorter than 12 words or 80 cleaned characters. These are heuristic overlapping units: mixed clauses, headers, short clauses and cross-page continuity require annotation. Exact duplicate units count once for topic discovery, with all occurrences retained privately.

## Compared options

| Method | Why consider it | Main limitation | Implemented |
| --- | --- | --- | --- |
| TF-IDF + LSA + KMeans | Cheap, interpretable lexical baseline | Template vocabulary dominates | Yes |
| TF-IDF + LSA + average-link cosine hierarchy | No spherical cluster assumption | Chain/large-cluster and singleton effects | Yes |
| TF-IDF + NMF | Additive, interpretable topics for mixed content | Lexical/form boilerplate | Yes, documents and units |
| Multilingual E5 + KMeans | Compact multilingual semantic baseline; full-text windows | Fixed k and issuer/template confounding | Yes, documents and units |
| Multilingual E5 + cosine hierarchy | Inspect nesting and unusual instruments | Outliers can dominate apparent quality | Yes, documents |
| E5 + HDBSCAN | Variable cluster count, explicit noise | Excludes many heterogeneous/rare units | Yes, documents and units |
| E5 + UMAP(5D) + HDBSCAN | BERTopic-style density exploration | Projection and density parameters affect topics | Yes, documents; no claim to run the BERTopic package |
| E5 + class TF-IDF descriptions | Human-readable terms for semantic groups | Terms describe a cluster, not legal truth | Yes, units |
| BGE-M3 or E5-base/large | Candidate stronger embeddings / longer input alternative | Must benchmark local quality and cost; longer context does not resolve mixed themes | Considered, not run |
| LLM-generated clusters / translated foreign legal benchmarks | Flexible naming and domain inspiration | Reproducibility, hallucinations, jurisdiction shift and annotation still needed | Not used to manufacture labels |

[E5 model card](https://huggingface.co/intfloat/multilingual-e5-small) specifies the clustering prefix and model limits. [BGE-M3](https://huggingface.co/BAAI/bge-m3) is an alternative for multilingual long inputs. [Sentence Transformers clustering](https://sbert.net/examples/sentence_transformer/applications/clustering/README.html), [BERTopic's algorithm](https://maartengr.github.io/BERTopic/algorithm/algorithm.html), and [scikit-learn's clustering guide](https://scikit-learn.org/stable/modules/clustering.html) motivate the comparisons.

## Metrics and selection

For documents, vary k in 3,4,5,6,8,10,12; density minimum sizes in 5,8,12,16; NMF components in 4,6,8,10,12. Evaluate all methods in the same original masked E5 space (LSA space in lexical-only mode), using cosine silhouette. The 2D PCA plot is only for navigation. E5-based methods have a representation advantage in this common space; it is a structural comparison, not neutral legal ground truth.

Report cluster sizes, largest-cluster share, coverage/noise, issuer adjusted mutual information, and extraction-route AMI. Five refits on 80% subsamples report mean ARI among shared non-noise members, conditional on the feature representation; TF-IDF/SVD/E5/UMAP preprocessing is not refit on each subsample. NMF stability is not estimated. Large ARI can reflect stable templates, and density stability can look high by excluding hard observations.

The provisional document candidate maximizes common-space silhouette subject to coverage >=80%, >=2 clusters and minimum observed cluster size >=3. This is a transparent discovery heuristic, **not an optimized legal taxonomy**. The size grid is bounded: a winner at its largest k is not evidence of the true number of themes. For units, compare KMeans k=8,12,16,20,24 and HDBSCAN minimum sizes 10,20,35, reporting the same silhouette/coverage caveat; select provisionally among coverage >=80%. Unit stability has not been estimated.

Character 3–5gram cosine >=0.94 supplies connected template families, then shared cases union those groups for future evaluation. Single-link chains can be larger than pairwise-near duplicates. These groups are an evaluation/sampling precaution, not ground-truth identity. No train/test split or supervised accuracy is claimed in this discovery study.

## Recommended next evaluation

Create a legally annotated Portuguese pilot, covering every case/template family and rare issuer/product. Use two independent annotators and adjudication. Label paragraph spans with multiple neutral themes; label product/instrument separately; distinguish what is stated, what is contradicted, what is missing from available text and what needs external evidence. Retain unknown and partial-text states.

Freeze a held-out group split before model choice; all documents from a shared case/template group stay on one side. Report per-label support, precision/recall/F1, macro/micro F1, span overlap, abstention coverage, and explanation/source quality. Confidence intervals should resample groups rather than duplicated paragraphs. Keep rare classes separate for review when they lack enough independent positives; do not promise a fixed sample size guarantees reliability.

Foreign contract datasets help with task design, not ready-to-use Brazilian labels: [CUAD](https://www.atticusprojectai.org/cuad/), [ContractNLI](https://arxiv.org/abs/2110.01799), [LexGLUE](https://aclanthology.org/2022.acl-long.297/) and [Chilean consumer-clause research](https://arxiv.org/abs/2502.00865) cover different languages, instruments and jurisdictions.
