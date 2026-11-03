"""Compare local discovery methods; scores measure structure, never legal correctness."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import NMF, PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from .corpus import write_json
from .text import STOPWORDS

DEFAULT_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"


def token_windows(ids, window=256, overlap=32):
    """Cover the complete token sequence, including the last token of long contracts."""
    if not 0 <= overlap < window <= 450:
        raise ValueError("invalid token window or overlap")
    for start in range(0, max(1, len(ids)), window - overlap):
        yield ids[start : start + window]
        if start + window >= len(ids):
            break


def leakage_groups(template_labels, case_groups):
    """Union documents sharing a template or case, for future grouped evaluation."""
    adjacency = np.eye(len(template_labels), dtype=bool)
    for i in range(len(template_labels)):
        for j in range(i):
            shared_case = bool(case_groups[i]) and case_groups[i] == case_groups[j]
            if template_labels[i] == template_labels[j] or shared_case:
                adjacency[i, j] = adjacency[j, i] = True
    return connected_components(adjacency, directed=False)[1]


def semantic_vectors(
    texts,
    output,
    *,
    model_name=DEFAULT_MODEL,
    model_revision=None,
    window=256,
    overlap=32,
    batch_size=32,
):
    """Encode every token window; query prefix is required for E5 clustering, per its card."""
    import torch
    from sentence_transformers import SentenceTransformer

    if not 0 <= overlap < window <= 450:
        raise ValueError("invalid token window or overlap")
    torch.set_num_threads(8)
    started = time.monotonic()
    requested_revision = model_revision or (
        DEFAULT_REVISION if model_name == DEFAULT_MODEL else None
    )
    model = SentenceTransformer(
        model_name, revision=requested_revision, device="cpu", trust_remote_code=False
    )
    revision = getattr(model[0].auto_model.config, "_commit_hash", None)
    if revision is None:
        from huggingface_hub import try_to_load_from_cache

        cached_config = try_to_load_from_cache(
            model_name, "config.json", revision=requested_revision
        )
        if isinstance(cached_config, str):
            revision = Path(cached_config).parent.name
    config = {
        "model": model_name,
        "revision": revision,
        "window": window,
        "overlap": overlap,
        "prefix": "query: ",
        "pooling": "length-weighted mean of normalized chunks then L2",
        "sentence_transformers": importlib.metadata.version("sentence-transformers"),
    }
    signature = hashlib.sha256(json.dumps([config, texts], ensure_ascii=False).encode()).hexdigest()
    cache = output / "embeddings.npy"
    info = output / "embedding_config.json"
    if (
        cache.exists()
        and info.exists()
        and json.loads(info.read_text()).get("signature") == signature
    ):
        return np.load(cache), json.loads(info.read_text())
    chunks, memberships, lengths = [], [], []
    for i, text in enumerate(texts):
        ids = model.tokenizer.encode(text, add_special_tokens=False, truncation=False)
        for part in token_windows(ids, window, overlap):
            chunks.append("query: " + model.tokenizer.decode(part, skip_special_tokens=True))
            memberships.append(i)
            lengths.append(max(1, len(part)))
    vectors = model.encode(
        chunks,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    result = np.zeros((len(texts), vectors.shape[1]), dtype=np.float32)
    weights = np.zeros(len(texts))
    for vector, i, weight in zip(vectors, memberships, lengths):
        result[i] += vector * weight
        weights[i] += weight
    result = normalize(result / weights[:, None]).astype(np.float32)
    np.save(cache, result)
    metadata = {
        **config,
        "signature": signature,
        "chunks": len(chunks),
        "documents": len(texts),
        "elapsed_seconds": time.monotonic() - started,
        "long_document_policy": "no first-page/first-512-token truncation",
    }
    write_json(info, metadata)
    return result, metadata


def estimator(method, k, seed=42):
    if method in {"tfidf_kmeans", "issuer_tfidf_kmeans", "e5_kmeans"}:
        return KMeans(n_clusters=k, random_state=seed, n_init=15)
    if method in {"tfidf_agglomerative", "e5_agglomerative"}:
        return AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
    if method == "e5_hdbscan":
        return HDBSCAN(
            min_cluster_size=k, min_samples=3, metric="cosine", algorithm="brute", copy=True
        )
    if method == "umap_hdbscan":
        return HDBSCAN(min_cluster_size=k, min_samples=3, metric="euclidean", copy=True)
    raise ValueError(method)


def metrics(labels, evaluation, issuers, routes):
    labels = np.asarray(labels)
    kept = labels >= 0
    groups = Counter(labels[kept].tolist())
    score = None
    if 1 < len(groups) < int(kept.sum()):
        score = float(silhouette_score(evaluation[kept], labels[kept], metric="cosine"))
    return {
        "clusters": len(groups),
        "noise": int((~kept).sum()),
        "coverage": float(kept.mean()),
        "silhouette_common_space": score,
        "min_cluster_size": min(groups.values()) if groups else 0,
        "max_cluster_fraction": max(groups.values()) / len(labels) if groups else 0,
        "issuer_ami": float(adjusted_mutual_info_score(issuers, labels)),
        "extraction_route_ami": float(adjusted_mutual_info_score(routes, labels)),
    }


def stability(method, features, k, labels, *, repeats=5):
    """Refit on 80% subsets; ARI on shared non-noise members, plus coverage in metrics."""
    values = []
    for seed in range(repeats):
        index = np.sort(
            np.random.default_rng(seed + 100).choice(
                len(labels), int(0.8 * len(labels)), replace=False
            )
        )
        subset = estimator(method, k, seed + 100).fit_predict(features[index])
        valid = (labels[index] >= 0) & (subset >= 0)
        if valid.sum() >= 10 and len(set(subset[valid])) > 1 and len(set(labels[index][valid])) > 1:
            values.append(float(adjusted_rand_score(labels[index][valid], subset[valid])))
    return float(np.mean(values)) if values else None


def descriptions(texts, labels, records, *, count=10):
    vectorizer = TfidfVectorizer(
        stop_words=STOPWORDS,
        strip_accents="unicode",
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
        max_features=20000,
    )
    matrix = vectorizer.fit_transform(texts)
    terms = np.asarray(vectorizer.get_feature_names_out())
    result = []
    for label in sorted(set(labels)):
        index = np.flatnonzero(labels == label)
        average = np.asarray(matrix[index].mean(axis=0)).ravel()
        top = terms[np.argsort(average)[-count:][::-1]].tolist()
        # Cosine to the cluster's lexical centroid: actual documents, not generated summaries.
        distances = cosine_similarity(matrix[index], average.reshape(1, -1)).ravel()
        chosen = index[np.argsort(distances)[-3:][::-1]]
        result.append(
            {
                "cluster": int(label),
                "documents": len(index),
                "terms": top,
                "institutions": dict(Counter(records[i]["institution"] for i in index)),
                "representatives": [
                    {"doc_id": records[i]["doc_id"], "snippet": texts[i][:700]} for i in chosen
                ],
            }
        )
    return result


def run_benchmark(
    records, output: Path, *, semantic=True, model_name=DEFAULT_MODEL, model_revision=None
):
    if len(records) < 20:
        raise ValueError("at least twenty distinct documents are needed for this benchmark grid")
    output.mkdir(parents=True, exist_ok=True)
    texts = [r["clean_text"] for r in records]
    issuers = [r["institution"] for r in records]
    routes = [r["method"] for r in records]
    vectorizer = TfidfVectorizer(
        stop_words=STOPWORDS,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=20000,
        sublinear_tf=True,
    )
    lexical = vectorizer.fit_transform(texts)
    dimensions = min(64, len(records) - 2, lexical.shape[1] - 1)
    lsa = normalize(TruncatedSVD(dimensions, random_state=42).fit_transform(lexical))
    issuer_matrix = TfidfVectorizer(
        stop_words=STOPWORDS,
        strip_accents="unicode",
        min_df=2,
        max_features=20000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    ).fit_transform([r["issuer_preserved_text"] for r in records])
    issuer_lsa = normalize(TruncatedSVD(dimensions, random_state=42).fit_transform(issuer_matrix))
    char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=40000
    ).fit_transform(texts)
    similarity = cosine_similarity(char)
    count, families = connected_components(similarity >= 0.94, directed=False)
    evaluation_groups = leakage_groups(families, [r.get("case_group") for r in records])
    write_json(
        output / "near_duplicate_groups.json",
        {
            "threshold": 0.94,
            "groups": count,
            "policy": "conservative connected template families for grouped evaluation; not discarded",
            "assignments": {r["doc_id"]: int(group) for r, group in zip(records, families)},
            "case_and_template_groups": len(set(evaluation_groups)),
            "evaluation_assignments": {
                r["doc_id"]: int(g) for r, g in zip(records, evaluation_groups)
            },
        },
    )
    embedding_metadata = None
    if semantic:
        dense, embedding_metadata = semantic_vectors(
            texts, output, model_name=model_name, model_revision=model_revision
        )
        evaluation = dense
    else:
        dense, evaluation = None, lsa
    spaces = {"tfidf_kmeans": lsa, "tfidf_agglomerative": lsa, "issuer_tfidf_kmeans": issuer_lsa}
    if semantic:
        spaces.update(e5_kmeans=dense, e5_agglomerative=dense, e5_hdbscan=dense)
        import umap

        reduced = umap.UMAP(
            n_components=5, n_neighbors=15, min_dist=0, metric="cosine", random_state=42, n_jobs=1
        ).fit_transform(dense)
        spaces["umap_hdbscan"] = reduced
    results, assignments = [], {}
    with threadpool_limits(limits=8):
        for method, features in spaces.items():
            grid = [5, 8, 12, 16] if "hdbscan" in method else [3, 4, 5, 6, 8, 10, 12]
            for k in grid:
                if k >= len(records):
                    continue
                begin = time.monotonic()
                labels = estimator(method, k).fit_predict(features)
                row = {
                    "method": method,
                    "parameter": k,
                    **metrics(labels, evaluation, issuers, routes),
                    "evaluation_space": "e5_masked_original" if semantic else "tfidf_lsa",
                }
                name = f"{method}_{k}"
                assignments[name] = labels
                row["subsample_stability_ari"] = stability(method, features, k, labels)
                row["seconds"] = time.monotonic() - begin
                results.append(row)
                print(json.dumps(row), flush=True)
        # NMF topics remain useful when documents contain multiple clause themes.
        for k in [4, 6, 8, 10, 12]:
            model = NMF(n_components=k, init="nndsvda", max_iter=400, random_state=42)
            weights = model.fit_transform(lexical)
            labels = weights.argmax(axis=1)
            row = {
                "method": "tfidf_nmf",
                "parameter": k,
                **metrics(labels, evaluation, issuers, routes),
                "evaluation_space": "e5_masked_original" if semantic else "tfidf_lsa",
                "subsample_stability_ari": None,
                "seconds": None,
            }
            results.append(row)
            assignments[f"tfidf_nmf_{k}"] = labels
    table = pd.DataFrame(results)
    table.to_csv(output / "benchmark.csv", index=False)
    write_json(output / "benchmark.json", results)
    valid = [
        r
        for r in results
        if r["coverage"] >= 0.8
        and r["clusters"] >= 2
        and r["min_cluster_size"] >= 3
        and r["silhouette_common_space"] is not None
    ]
    # A provisional geometry-based candidate, explicitly not the final legal taxonomy.
    best = max(
        valid or [r for r in results if r["silhouette_common_space"] is not None],
        key=lambda r: r["silhouette_common_space"],
    )
    name = f"{best['method']}_{best['parameter']}"
    labels = assignments[name]
    write_json(
        output / "selection.json",
        {
            "candidate": best,
            "rule": "highest common-space silhouette among coverage>=.8, >=2 clusters and min size>=3; analyst review still required",
            "warning": "not legal accuracy; issuer and template effects are reported as confounders",
            "embedding": embedding_metadata,
        },
    )
    write_json(output / "clusters.json", descriptions(texts, labels, records))
    rows = [
        {
            "doc_id": r["doc_id"],
            "institution": r["institution"],
            "technical_route": r["method"],
            "template_group": int(family),
            "case_group": r.get("case_group"),
            "evaluation_group": int(evaluation_groups[i]),
            **{name: int(v[i]) for name, v in assignments.items()},
        }
        for i, (r, family) in enumerate(zip(records, families))
    ]
    pd.DataFrame(rows).to_csv(output / "assignments.csv", index=False)
    # Every point's projection is visual-only; no quality metric uses this 2-D space.
    projection = PCA(2, random_state=42).fit_transform(evaluation)
    from .reporting import dashboard

    dashboard(records, projection, labels, results, output / "clusters.html")
    write_json(
        output / "run_summary.json",
        {
            "documents": len(records),
            "experiments": len(results),
            "template_groups": count,
            "selected_candidate": name,
            "versions": {
                p: importlib.metadata.version(p)
                for p in ["numpy", "scikit-learn", "scipy", "pymupdf"]
            },
            "semantic": semantic,
            "gold_labels": False,
        },
    )
    return results, assignments, evaluation


def clause_topics(records, output, *, topics=12):
    """NMF over deduplicated heuristic clause units; retain occurrence provenance privately."""
    unique = {}
    for record in records:
        for unit in record["units"]:
            key = hashlib.sha256(unit["text"].encode()).hexdigest()
            item = unique.setdefault(key, {"text": unit["text"], "occurrences": []})
            item["occurrences"].append(
                {
                    "doc_id": record["doc_id"],
                    "processed_page": unit["processed_page"],
                    "unit_id": unit["unit_id"],
                }
            )
    units = list(unique.values())
    vectorizer = TfidfVectorizer(
        stop_words=STOPWORDS,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        max_features=20000,
    )
    matrix = vectorizer.fit_transform([r["text"] for r in units])
    topics = min(topics, matrix.shape[0] - 1, matrix.shape[1] - 1)
    with threadpool_limits(8):
        nmf = NMF(n_components=topics, init="nndsvda", max_iter=400, random_state=42)
        weights = nmf.fit_transform(matrix)
    terms = np.asarray(vectorizer.get_feature_names_out())
    groups = []
    for i in range(topics):
        chosen = np.argsort(weights[:, i])[-4:][::-1]
        groups.append(
            {
                "topic": i,
                "terms": terms[np.argsort(nmf.components_[i])[-15:][::-1]].tolist(),
                "dominant_units": int((weights.argmax(axis=1) == i).sum()),
                "examples": [units[j] for j in chosen],
            }
        )
    write_json(
        output / "clause_topics.json",
        {
            "unique_units": len(units),
            "topics": groups,
            "warning": "heuristic spans and exploratory topics, not annotated clause classes",
        },
    )
    return groups
