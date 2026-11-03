from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import audit_taxonomy
from .benchmark import clause_topics, run_benchmark
from .clauses import discover_clauses
from .corpus import build_corpus


def main(argv=None):
    parser = argparse.ArgumentParser(description="Local processed-contract clustering research")
    sub = parser.add_subparsers(dest="command", required=True)
    inventory = sub.add_parser("inventory")
    inventory.add_argument("--input", type=Path, required=True)
    inventory.add_argument("--output", type=Path, required=True)
    inventory.add_argument("--metadata", type=Path)
    run = sub.add_parser("benchmark")
    run.add_argument("--corpus", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--lexical-only", action="store_true")
    run.add_argument("--model", default="intfloat/multilingual-e5-small")
    run.add_argument("--revision", help="Model revision override; default E5-small is pinned")
    clauses = sub.add_parser("clauses")
    clauses.add_argument("--corpus", type=Path, required=True)
    clauses.add_argument("--output", type=Path, required=True)
    clauses.add_argument("--model", default="intfloat/multilingual-e5-small")
    clauses.add_argument("--revision", help="Model revision override; default E5-small is pinned")
    audit = sub.add_parser("audit")
    audit.add_argument("--corpus", type=Path, required=True)
    audit.add_argument("--taxonomy", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "inventory":
        _, summary = build_corpus(args.input, args.output, args.metadata)
        print(json.dumps(summary, ensure_ascii=False))
    else:
        records = [
            json.loads(line)
            for line in args.corpus.read_text(encoding="utf-8").splitlines()
            if line
        ]
        if args.command == "audit":
            audit_taxonomy(records, args.taxonomy, args.output)
        elif args.command == "clauses":
            discover_clauses(
                records, args.output, model_name=args.model, model_revision=args.revision
            )
        else:
            run_benchmark(
                records,
                args.output,
                semantic=not args.lexical_only,
                model_name=args.model,
                model_revision=args.revision,
            )
            clause_topics(records, args.output)
    return 0
