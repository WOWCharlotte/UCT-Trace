"""Cascade entry point for attack behavior judging.

The candidate-audit mode is deliberately usable without an API key.  Later
stages add the external judge while preserving this independent JSONL output.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from attack_judge_support import (
    FORMULA_VERSION,
    JudgeInput,
    agentdojo_input,
    audit_counts,
    injecagent_input,
    select_candidates,
    stable_input_hash,
)


def read_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_audit_rows(rows: list[dict[str, Any]], dataset: str, mode: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates, skipped = select_candidates(rows, mode)
    output = []
    for source_row in candidates + skipped:
        if dataset == "injecagent":
            context = injecagent_input(source_row)
        elif dataset == "agentdojo":
            trajectory = source_row.get("trajectory") or {}
            context = agentdojo_input(source_row, trajectory)
        else:
            raise ValueError("dataset must be injecagent or agentdojo")
        annotated = dict(source_row)
        annotated["judge_input"] = context.as_dict()
        annotated["input_hash"] = stable_input_hash(context)
        if not annotated["attack_attribution_trigger"]:
            annotated["cascade_status"] = "not_judged_by_cascade"
        else:
            annotated["cascade_status"] = "candidate_pending_judge"
        output.append(annotated)
    summary = {
        "schema_version": 1,
        "formula_version": FORMULA_VERSION,
        "dataset": dataset,
        "trigger_rule": mode,
        **audit_counts(len(rows), len(candidates)),
        "candidate_pending_count": len(candidates),
    }
    return output, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit attack attribution candidates before LLM judging.")
    parser.add_argument("--input", required=True, help="Existing attribution JSONL")
    parser.add_argument("--output", required=True, help="Independent candidate/audit JSONL")
    parser.add_argument("--summary", required=True, help="Audit summary JSON")
    parser.add_argument("--dataset", choices=("injecagent", "agentdojo"), required=True)
    parser.add_argument("--trigger", choices=("shapley", "attention", "both"), default="shapley")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, summary = build_audit_rows(read_jsonl(args.input), args.dataset, args.trigger)
    write_jsonl(args.output, rows)
    with open(args.summary, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
