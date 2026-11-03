"""Complement document discovery with deduplicated semantic clause/window discovery."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from threadpoolctl import threadpool_limits

from .benchmark import semantic_vectors
from .corpus import write_json
from .text import STOPWORDS


def distinct_units(records):
    unique = {}
    for record in records:
        for unit in record["units"]:
            key = hashlib.sha256(unit["text"].encode()).hexdigest()
            item = unique.setdefault(
                key, {"unit_hash": key, "text": unit["text"], "occurrences": []}
            )
            item["occurrences"].append(
                {
                    "doc_id": record["doc_id"],
                    "unit_id": unit["unit_id"],
                    "processed_page": unit["processed_page"],
                    "institution": record["institution"],
                }
            )
    return list(unique.values())


def discover_clauses(
    records, output: Path, model_name="intfloat/multilingual-e5-small", model_revision=None
):
    output.mkdir(parents=True, exist_ok=True)
    units = distinct_units(records)
    if len(units) < 40:
        raise ValueError("at least forty distinct clause/windows are required")
    texts = [u["text"] for u in units]
    vectors, embedding = semantic_vectors(
        texts, output, model_name=model_name, model_revision=model_revision
    )
    results, assignments = [], {}
    with threadpool_limits(8):
        for k in [8, 12, 16, 20, 24]:
            labels = KMeans(n_clusters=k, n_init=15, random_state=42).fit_predict(vectors)
            assignments[f"kmeans_{k}"] = labels
        for size in [10, 20, 35]:
            labels = HDBSCAN(
                min_cluster_size=size, min_samples=5, metric="cosine", algorithm="brute", copy=True
            ).fit_predict(vectors)
            assignments[f"hdbscan_{size}"] = labels
        for name, labels in assignments.items():
            kept = labels >= 0
            counts = Counter(labels[kept].tolist())
            score = None
            if 1 < len(counts) < kept.sum():
                score = float(silhouette_score(vectors[kept], labels[kept], metric="cosine"))
            results.append(
                {
                    "method": name,
                    "clusters": len(counts),
                    "coverage": float(kept.mean()),
                    "noise": int((~kept).sum()),
                    "silhouette": score,
                    "min_size": min(counts.values()) if counts else 0,
                }
            )
    eligible = [r for r in results if r["coverage"] >= 0.8 and r["silhouette"] is not None]
    best = max(eligible, key=lambda r: r["silhouette"])
    labels = assignments[best["method"]]
    # Class TF-IDF over discovered groups: descriptive labels, never generated legal verdicts.
    group_ids = sorted(set(labels))
    vectorizer = CountVectorizer(
        stop_words=STOPWORDS,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_features=20000,
    )
    matrix = vectorizer.fit_transform(texts)
    grouped = np.vstack([np.asarray(matrix[labels == i].sum(axis=0)).ravel() for i in group_ids])
    average_words = grouped.sum(axis=1).mean()
    idf = np.log1p(average_words / np.maximum(1, grouped.sum(axis=0)))
    tfidf = grouped / np.maximum(1, grouped.sum(axis=1))[:, None] * idf[None, :]
    vocabulary = np.asarray(vectorizer.get_feature_names_out())
    clusters = []
    for row, group in enumerate(group_ids):
        index = np.flatnonzero(labels == group)
        centroid = vectors[index].mean(axis=0, keepdims=True)
        similarities = cosine_similarity(vectors[index], centroid).ravel()
        chosen = index[np.argsort(similarities)[-4:][::-1]]
        doc_ids = {o["doc_id"] for j in index for o in units[j]["occurrences"]}
        issuer_counts = Counter(o["institution"] for j in index for o in units[j]["occurrences"])
        clusters.append(
            {
                "cluster": int(group),
                "unique_units": len(index),
                "distinct_documents": len(doc_ids),
                "institution_occurrences": dict(issuer_counts),
                "terms": vocabulary[np.argsort(tfidf[row])[-15:][::-1]].tolist(),
                "representatives": [units[j] for j in chosen],
            }
        )
    write_json(output / "clause_clusters.json", clusters)
    write_json(
        output / "clause_benchmark.json",
        {
            "units": len(units),
            "experiments": results,
            "candidate": best,
            "embedding": embedding,
            "warning": "Heuristic overlapping windows, not gold clauses. No legal accuracy or out-of-sample validation.",
        },
    )
    write_json(
        output / "clause_assignments.json",
        [
            {"unit_hash": u["unit_hash"], "cluster": int(label), "occurrences": u["occurrences"]}
            for u, label in zip(units, labels)
        ],
    )
    return results
