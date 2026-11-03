"""Standalone local HTML; no text or document is sent to a hosted dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px


def dashboard(records, points, labels, results, path):
    frame = pd.DataFrame(
        {
            "x": points[:, 0],
            "y": points[:, 1],
            "cluster": labels.astype(str),
            "doc_id": [r["doc_id"] for r in records],
            "institution": [r["institution"] for r in records],
            "route": [r["method"] for r in records],
            "pages": [r["pages"] for r in records],
        }
    )
    figure = px.scatter(
        frame,
        x="x",
        y="y",
        color="cluster",
        hover_data=["doc_id", "institution", "route", "pages"],
        title="Contract discovery: projection for exploration only",
    )
    figure.update_traces(marker={"size": 9, "opacity": 0.8})
    figure.update_layout(template="plotly_white", height=680)
    table = pd.DataFrame(results).sort_values("silhouette_common_space", ascending=False)
    html = "<!doctype html><meta charset='utf-8'><title>Contract clustering audit</title>"
    html += "<style>body{font:16px system-ui;max-width:1400px;margin:30px auto;padding:20px}table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border:1px solid #ddd}aside{background:#fff3cd;padding:20px}</style>"
    html += "<h1>Contract clustering audit</h1><aside>Exploratory clusters are not legal labels or accuracy. "
    html += "Identifiers and known institution aliases are masked; extraction folders are not features; "
    html += "their association with clusters is audited. No point's quality is scored in this 2-D projection.</aside>"
    html += figure.to_html(full_html=False, include_plotlyjs=True)
    html += "<h2>Measured experiments</h2>" + table.round(4).to_html(index=False, escape=True)
    html += "<p>Local standalone artifact. Documents and texts remain on this machine.</p>"
    path.write_text(html, encoding="utf-8")
