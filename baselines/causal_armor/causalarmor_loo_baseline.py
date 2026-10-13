"""CausalArmor's tool-result LOO criterion adapted to fixed-action analysis.

This is an offline attribution baseline, not the CausalArmor online guardrail.
It neither sanitizes tool results nor regenerates agent actions.  The same
target model, prompt, and fixed action used by UCT-Trace are scored under
three contexts: full, without the original user request, and without the
entire polluted tool-result content.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS_ROOT = os.path.join(REPO_ROOT, "scripts")
for import_root in (REPO_ROOT, SCRIPTS_ROOT):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from agentdojo_attribution_experiment import (
    audit_corpus,
    attribution_dedup_key,
    case_id as agentdojo_case_id,
    clear_cuda_cache,
    embedding_layer,
    fact_attack_lengths,
    locate_source,
    mean_logprob_with_masked_regions,
    model_device,
    original_user_message,
    player_texts,
    read_target_id_file,
    render_selected_target,
    selected_target_kind,
)
from baselines.attention_tracker.utils import create_model, open_config
from injecagent_action_experiment import (
    action_dedup_key,
    build_injecagent_prompt_parts,
    build_tool_index,
    parse_case_ids,
    parse_action,
    read_json,
    read_jsonl,
    tokenize_text,
)


def ablate_text(text: str, start: int, end: int) -> str:
    """Remove exactly one located content region, preserving role wrappers."""
    if not 0 <= start < end <= len(text):
        raise ValueError(f"Invalid ablation bounds: {(start, end)}")
    return text[:start] + text[end:]


def loo_scores(full: float, without_user: float, without_tool: float, tau: float) -> dict[str, Any]:
    """Port the normalized LOO difference and margin rule from CausalArmor."""
    values = (full, without_user, without_tool, tau)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("LOO scores and tau must be finite")
    delta_user = full - without_user
    delta_tool = full - without_tool
    margin = delta_tool - delta_user
    return {
        "logprob_full": full,
        "logprob_without_user": without_user,
        "logprob_without_tool_result": without_tool,
        "delta_user": delta_user,
        "delta_tool_result": delta_tool,
        "dominance_margin": margin,
        "tau": tau,
        "predicted_attack_success": margin > -tau,
    }


def score_fixed_action(
    model: Any,
    prompt_ids: list[int] | tuple[int, ...],
    target_ids: list[int] | tuple[int, ...],
    *,
    full_forward_token_threshold: int,
    streaming_chunk_size: int,
    prefill_chunk_size: int,
) -> tuple[float, str]:
    """Reuse UCT-Trace's teacher-forced mean target-token log probability."""
    if not prompt_ids or not target_ids:
        raise ValueError("Prompt and fixed action must each contain tokens")
    device = model_device(model)
    prompt = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    target = torch.tensor([target_ids], device=device, dtype=torch.long)
    prompt_embeds = embedding_layer(model)(prompt).detach()
    prefer_full = (
        full_forward_token_threshold <= 0
        or len(prompt_ids) + len(target_ids) <= full_forward_token_threshold
    )
    try:
        value, strategy, _ = mean_logprob_with_masked_regions(
            model,
            prompt_embeds,
            target,
            {},
            (),
            prefer_full_forward=prefer_full,
            streaming_chunk_size=streaming_chunk_size,
            prefill_chunk_size=prefill_chunk_size,
        )
        return value, strategy
    finally:
        del prompt_embeds, prompt, target
        clear_cuda_cache()


def injecagent_variants(
    model: Any, source: dict[str, Any], tools: dict[str, dict[str, Any]],
    *, include_all_tools: bool = False,
) -> tuple[list[list[int]], list[int], str]:
    parts = build_injecagent_prompt_parts(source, tools, include_all_tools=include_all_tools)
    prompt = parts["prompt"]
    user = str(source["User Instruction"])
    tool = parts["tool_response"]
    auth_wrapper = parts["auth_text"]
    auth_start = prompt.find(auth_wrapper)
    if auth_start < 0 or not user or not tool:
        raise ValueError("Could not locate nonempty InjecAgent user and tool content")
    user_start = auth_start + len("<user>\n")
    if prompt[user_start:user_start + len(user)] != user:
        raise ValueError("InjecAgent user content does not match rendered prompt")
    observation_start = prompt.find("Observation: ", user_start + len(user))
    tool_start = prompt.find(tool, observation_start + len("Observation: ")) if observation_start >= 0 else -1
    if tool_start < 0:
        raise ValueError("Could not locate InjecAgent tool result after Observation")
    parsed = parse_action(source.get("output", ""))
    if parsed.kind == "invalid_action_parse" or not parsed.text:
        raise ValueError("InjecAgent fixed action is invalid")
    prompts = (
        prompt,
        ablate_text(prompt, user_start, user_start + len(user)),
        ablate_text(prompt, tool_start, tool_start + len(tool)),
    )
    return [tokenize_text(model, item) for item in prompts], tokenize_text(model, parsed.text), parsed.kind


def agentdojo_variants(model: Any, selected: Any) -> tuple[list[list[int]], list[int], str]:
    rendered = render_selected_target(model.tokenizer, selected)
    prompt = rendered.prompt
    user = str(original_user_message(selected.trajectory).get("content") or "")
    tool = str(selected.trajectory["messages"][selected.polluted_tool_index].get("content") or "")
    if not user or not tool:
        raise ValueError("AgentDojo user or polluted tool content is empty")
    user_start, user_end = locate_source(prompt, user)
    tool_start, tool_end = locate_source(prompt, tool, start=user_end)
    if not user_start < user_end <= tool_start < tool_end:
        raise ValueError("AgentDojo user and polluted tool spans overlap or are out of order")
    tokenizer = model.tokenizer
    encode = lambda value: list(tokenizer(value, add_special_tokens=False)["input_ids"])
    base_ids = encode(prompt)
    if base_ids != list(rendered.prompt_ids):
        raise ValueError("Retokenized AgentDojo prompt differs from UCT-Trace prompt IDs")
    prompts = (
        base_ids,
        encode(ablate_text(prompt, user_start, user_end)),
        encode(ablate_text(prompt, tool_start, tool_end)),
    )
    return [list(ids) for ids in prompts], list(rendered.target_ids), rendered.target_kind


def metrics(
    rows: list[dict[str, Any]], prediction_field: str, *, tau: float | None = None
) -> dict[str, Any]:
    def prediction(row: dict[str, Any]) -> bool:
        if tau is None:
            return bool(row[prediction_field])
        return float(row["dominance_margin"]) > -tau

    tp = sum(prediction(row) and row["gold_attack_success"] for row in rows)
    fp = sum(prediction(row) and not row["gold_attack_success"] for row in rows)
    tn = sum(not prediction(row) and not row["gold_attack_success"] for row in rows)
    fn = sum(not prediction(row) and row["gold_attack_success"] for row in rows)
    n = len(rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "count": n,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": (tp + tn) / n if n else 0.0,
        "precision": precision,
        "recall": recall,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def select_agentdojo_cases(cases: list[Any], args: argparse.Namespace) -> tuple[list[Any], dict[str, Any]]:
    """Apply the same raw-corpus filters, ordering, limit, and dedup as UCT-Trace."""
    before_filters = len(cases)
    if args.action_targets_only:
        cases = [case for case in cases if selected_target_kind(case) in {"tool_calls", "mixed"}]
    if args.fact_dominant:
        filtered = []
        for case in cases:
            fact_chars, attack_chars = fact_attack_lengths(case)
            if fact_chars < args.min_fact_chars or fact_chars > args.max_fact_chars:
                continue
            if fact_chars < args.fact_ratio * max(1, attack_chars):
                continue
            filtered.append(case)
        cases = filtered
        if args.rank_by_fact_ratio:
            cases = sorted(cases, key=lambda case: (
                fact_attack_lengths(case)[0] / max(1, fact_attack_lengths(case)[1]),
                fact_attack_lengths(case)[0],
            ), reverse=True)
    if args.target_id:
        cases = [case for case in cases if case.target_id == args.target_id]
        if not cases:
            raise ValueError(f"Unknown target id: {args.target_id}")
    if args.target_id_file:
        target_ids = read_target_id_file(args.target_id_file)
        if not target_ids:
            raise ValueError(f"Target id file is empty: {args.target_id_file}")
        cases = [case for case in cases if case.target_id in target_ids]
        if not cases:
            raise ValueError(f"No selected cases matched target ids from: {args.target_id_file}")
    filtered_count = len(cases)
    if args.limit is not None:
        cases = cases[:args.limit]
    pre_dedup_count = len(cases)
    unique_cases = []
    seen: set[tuple[Any, ...]] = set()
    for case in cases:
        key = attribution_dedup_key(case)
        if key not in seen:
            seen.add(key)
            unique_cases.append(case)
    return unique_cases, {
        "source": "raw_agentdojo_corpus",
        "before_filters": before_filters,
        "after_filters": filtered_count,
        "before_deduplication": pre_dedup_count,
        "selected_count": len(unique_cases),
        "duplicate_count": pre_dedup_count - len(unique_cases),
        "predicate": {
            "action_targets_only": args.action_targets_only,
            "fact_dominant": args.fact_dominant,
            "min_fact_chars": args.min_fact_chars,
            "max_fact_chars": args.max_fact_chars,
            "fact_ratio": args.fact_ratio,
            "fact_comparison": "FACT >= fact_ratio * max(1, ATTACK)",
            "rank_by_fact_ratio": args.rank_by_fact_ratio,
            "target_id": args.target_id,
            "target_id_file": args.target_id_file,
            "limit": args.limit,
        },
    }


def select_injecagent_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select fixed actions directly from generated-case JSONL files."""
    selected_ids = parse_case_ids(args.case_ids)
    candidates: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    for group, path in enumerate(args.input):
        rows = read_jsonl(path)
        source_counts[path] = len(rows)
        for identity, source in enumerate(rows):
            if args.eval_filter != "all" and source.get("eval") != args.eval_filter:
                continue
            if selected_ids is not None and identity not in selected_ids:
                continue
            candidates.append({
                "group": group, "case_id": identity, "sample_id": f"{group}:{identity}",
                "input_source": path, "source": source,
            })
    filtered_count = len(candidates)
    if args.limit is not None:
        candidates = candidates[:args.limit]
    pre_dedup_count = len(candidates)
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for candidate in candidates:
        parsed = parse_action(candidate["source"].get("output", ""))
        key = action_dedup_key(candidate["source"], parsed)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique, {
        "source": "raw_injecagent_jsonl",
        "input_counts": source_counts,
        "after_filters": filtered_count,
        "before_deduplication": pre_dedup_count,
        "selected_count": len(unique),
        "duplicate_count": pre_dedup_count - len(unique),
        "predicate": {
            "eval_filter": args.eval_filter,
            "case_ids": sorted(selected_ids) if selected_ids is not None else None,
            "limit": args.limit,
        },
    }


def read_comparison_rows(paths: list[str] | None, dataset: str) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    """Load optional UCT-Trace results for a post-hoc join only."""
    if not paths:
        return {}, None
    joined: dict[str, dict[str, Any]] = {}
    total = valid = 0
    for group, path in enumerate(paths):
        rows = read_jsonl(path)
        total += len(rows)
        for row in rows:
            if row.get("valid_for_stats") is not True:
                continue
            if dataset == "injecagent" and row.get("target_scope") != "full_action":
                continue
            identity = str(row.get("case_id") if dataset == "injecagent" else row.get("target_id") or "")
            if not identity:
                continue
            key = f"{group}:{identity}" if dataset == "injecagent" else identity
            if key in joined:
                raise ValueError(f"Duplicate comparison identity: {key}")
            joined[key] = row
            valid += 1
    return joined, {"paths": paths, "attempted_count": total, "valid_count": valid}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(
    rows: list[dict[str, Any]], *, dataset: str, model_config: str,
    corpus_audit: dict[str, Any], selection: dict[str, Any], tau: float,
    tau_grid: list[float], comparison_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    valid = [row for row in rows if row.get("valid_for_stats") is True]
    tool_kinds = {"tool_action"} if dataset == "injecagent" else {"tool_calls", "mixed"}
    tool_rows = [row for row in valid if row["target_kind"] in tool_kinds]
    times = [row["loo_time_seconds"] for row in valid]
    failed = [row for row in rows if row.get("valid_for_stats") is not True]
    paired = [row for row in valid if isinstance(row.get("shapley_attack_dominant"), bool)]
    result = {
        "schema_version": 3,
        "method": "causalarmor_tool_result_loo_offline_adaptation",
        "scope_note": "Detection-only fixed-action comparison; no privileged-action gate, sanitization, CoT masking, or regeneration.",
        "dataset": dataset,
        "model_config": model_config,
        "selection_source": "raw_dataset",
        "corpus_audit": corpus_audit,
        "selection": selection,
        "tau": tau,
        "selected_count": len(rows),
        "valid_count": len(valid),
        "failed_count": len(failed),
        "error_classes": dict(Counter(str(row.get("error", "UnknownError")).split(":", 1)[0] for row in failed)),
        "groups": {
            "suite": dict(Counter(str(row["suite_name"]) for row in valid if row.get("suite_name") is not None)),
            "security": dict(Counter(str(row.get("security")) for row in valid)),
            "target_kind": dict(Counter(str(row.get("target_kind")) for row in valid)),
        },
        "suite_group_note": "suite_name=null is excluded from suite counts.",
        "overall": metrics(valid, "predicted_attack_success"),
        "tool_action_subset": metrics(tool_rows, "predicted_attack_success"),
        "privileged_action_proxy_subset": metrics(tool_rows, "predicted_attack_success"),
        "privileged_action_proxy_note": "Tool-call targets are used as the available proxy; no benchmark-specific privileged-tool allowlist is available.",
        "tau_sweep": [
            {"tau": candidate, **metrics(valid, "predicted_attack_success", tau=candidate)}
            for candidate in tau_grid
        ],
        "runtime_seconds": {
            "mean": statistics.mean(times) if times else None,
            "median": statistics.median(times) if times else None,
        },
    }
    if comparison_audit is not None:
        result["posthoc_uct_trace_comparison"] = {
            "audit": comparison_audit,
            "overlap_count": len(paired),
            "causalarmor": metrics(paired, "predicted_attack_success"),
            "uct_trace": metrics(paired, "shapley_attack_dominant"),
            "disagreements": {
                "both_attack": sum(row["predicted_attack_success"] and row["shapley_attack_dominant"] for row in paired),
                "causalarmor_only": sum(row["predicted_attack_success"] and not row["shapley_attack_dominant"] for row in paired),
                "uct_trace_only": sum(not row["predicted_attack_success"] and row["shapley_attack_dominant"] for row in paired),
                "both_benign": sum(not row["predicted_attack_success"] and not row["shapley_attack_dominant"] for row in paired),
            },
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("injecagent", "agentdojo"), required=True)
    parser.add_argument(
        "--comparison-shapley", "--shapley-manifest", dest="comparison_shapley", nargs="+",
        help="Optional UCT-Trace JSONL used only for a post-hoc identity join; never controls CausalArmor selection.",
    )
    parser.add_argument("--model-config", required=True, help="Existing UCT-Trace model configuration JSON")
    parser.add_argument("--output-dir", required=True, help="New directory; existing attribution files are untouched")
    parser.add_argument("--input", nargs="+", help="One or more raw InjecAgent generated-case JSONL files")
    parser.add_argument("--tools", default="data/injecagent/injecagent_data/tools.json")
    parser.add_argument("--eval-filter", choices=("all", "succ", "unsucc", "invalid"), default="all")
    parser.add_argument("--case-ids", help="Comma-separated original InjecAgent row indices")
    parser.add_argument("--include-all-tools", action="store_true")
    parser.add_argument("--input-root", help="AgentDojo trajectory root")
    parser.add_argument("--fact-dominant", action="store_true")
    parser.add_argument("--action-targets-only", action="store_true")
    parser.add_argument("--min-fact-chars", type=int, default=500)
    parser.add_argument("--max-fact-chars", type=int, default=100000)
    parser.add_argument("--fact-ratio", type=float, default=1.0)
    parser.add_argument("--rank-by-fact-ratio", action="store_true")
    parser.add_argument("--target-id")
    parser.add_argument("--target-id-file")
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument(
        "--tau-grid", default="-1,-0.4,0,0.4,1",
        help="Comma-separated margins evaluated from the same cached LOO scores.",
    )
    parser.add_argument("--limit", type=int, help="Limit raw-input candidates for a small run")
    parser.add_argument(
        "--use-cache", action="store_true",
        help="Reuse valid rows in the output JSONL and checkpoint every completed sample.",
    )
    parser.add_argument("--full-forward-token-threshold", type=int, default=1024)
    parser.add_argument("--streaming-chunk-size", type=int, default=16)
    parser.add_argument("--prefill-chunk-size", type=int, default=256)
    args = parser.parse_args()
    if args.dataset == "injecagent" and not args.input:
        parser.error("--input is required for InjecAgent")
    if args.dataset == "agentdojo" and not args.input_root:
        parser.error("--input-root is required for AgentDojo")
    if args.dataset == "agentdojo" and args.comparison_shapley and len(args.comparison_shapley) != 1:
        parser.error("AgentDojo accepts at most one comparison Shapley manifest")
    if args.dataset == "injecagent" and args.comparison_shapley and len(args.comparison_shapley) != len(args.input):
        parser.error("InjecAgent comparison manifests, when provided, must align one-to-one with --input")
    if not math.isfinite(args.tau):
        parser.error("--tau must be finite")
    try:
        args.tau_grid = [float(value.strip()) for value in args.tau_grid.split(",") if value.strip()]
    except ValueError:
        parser.error("--tau-grid must contain comma-separated finite numbers")
    if not args.tau_grid or not all(math.isfinite(value) for value in args.tau_grid):
        parser.error("--tau-grid must contain comma-separated finite numbers")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.min_fact_chars < 0 or args.max_fact_chars < args.min_fact_chars or args.fact_ratio < 0:
        parser.error("AgentDojo FACT length settings must be nonnegative and max >= min")
    if args.streaming_chunk_size <= 0 or args.prefill_chunk_size <= 0:
        parser.error("chunk sizes must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.dataset == "injecagent":
        candidates, selection = select_injecagent_rows(args)
        corpus_audit = {
            "input_sources": args.input,
            "candidate_count": sum(selection["input_counts"].values()),
        }
        tool_index = build_tool_index(read_json(args.tools))
    else:
        raw_cases, _, corpus_audit = audit_corpus(args.input_root)
        selected_cases, selection = select_agentdojo_cases(raw_cases, args)
        candidates = [{
            "sample_id": case.target_id,
            "target_id": case.target_id,
            "input_source": args.input_root,
            "selected_case": case,
        } for case in selected_cases]
        tool_index = None
    comparison_rows, comparison_audit = read_comparison_rows(args.comparison_shapley, args.dataset)

    output_dir = Path(args.output_dir)
    output_path = output_dir / "results.causalarmor_loo.jsonl"
    selected_ids = {candidate["sample_id"] for candidate in candidates}
    cached_rows = read_jsonl(str(output_path)) if args.use_cache and output_path.exists() else []
    output = [
        row for row in cached_rows
        if row.get("valid_for_stats") is True
        and str(row.get("sample_id")) in selected_ids
    ]
    for row in output:
        row["schema_version"] = 3
        row["selection_source"] = "raw_dataset"
        row.pop("shapley_manifest", None)
        row.update(loo_scores(
            float(row["logprob_full"]), float(row["logprob_without_user"]),
            float(row["logprob_without_tool_result"]), args.tau,
        ))
        comparison = comparison_rows.get(str(row["sample_id"]))
        if comparison is not None:
            row["shapley_attack_dominant"] = comparison.get(
                "shapley_attack_dominant", comparison.get("attack_dominant")
            )
        else:
            row.pop("shapley_attack_dominant", None)
    completed_ids = {str(row["sample_id"]) for row in output}
    pending = [candidate for candidate in candidates if candidate["sample_id"] not in completed_ids]
    model = create_model(open_config(args.model_config)) if pending else None
    print(f"CausalArmor LOO: {len(output)} cached, {len(pending)} pending, {len(candidates)} selected from raw input")

    for pending_index, candidate in enumerate(pending, start=1):
        sample_id = candidate["sample_id"]
        print(f"[{pending_index}/{len(pending)}] {sample_id}", flush=True)
        record: dict[str, Any] = {
            "schema_version": 3,
            "dataset": args.dataset,
            "method": "causalarmor_tool_result_loo_offline_adaptation",
            "selection_source": "raw_dataset",
            "sample_id": sample_id,
            "input_source": candidate["input_source"],
            "valid_for_stats": False,
            "ablation_unit": "original_user_content_vs_entire_polluted_tool_result_content",
        }
        comparison = comparison_rows.get(sample_id)
        if comparison is not None:
            prediction = comparison.get("shapley_attack_dominant", comparison.get("attack_dominant"))
            if isinstance(prediction, bool):
                record["shapley_attack_dominant"] = prediction
        started = time.perf_counter()
        try:
            if args.dataset == "injecagent":
                source = candidate["source"]
                identity = candidate["case_id"]
                parsed = parse_action(source.get("output", ""))
                record.update({
                    "case_id": identity,
                    "target_scope": "full_action",
                    "target_text": parsed.text,
                    "target_kind": parsed.kind,
                    "action_kind": parsed.kind,
                    "action_text": parsed.text,
                    "action_tool": parsed.tool_name,
                    "action_input": parsed.action_input,
                    "eval": source.get("eval"),
                    "attack_type": source.get("Attack Type"),
                    "gold_attack_success": source.get("eval") == "succ",
                    "User Instruction": source.get("User Instruction"),
                    "Tool Response": source.get("Tool Response"),
                    "Attacker Instruction": source.get("Attacker Instruction"),
                    "output": source.get("output"),
                })
                if source.get("eval") not in {"succ", "unsucc"}:
                    raise ValueError("InjecAgent row has no success/failure label")
                prompts, target, kind = injecagent_variants(
                    model, source, tool_index, include_all_tools=args.include_all_tools,
                )
            else:
                selected_case = candidate["selected_case"]
                trajectory = selected_case.trajectory
                security = trajectory.get("security")
                record.update({
                    "target_id": selected_case.target_id,
                    "case_id": agentdojo_case_id(trajectory),
                    "source_path": selected_case.source_path,
                    "target_scope": "next_assistant_message",
                    "suite_name": trajectory.get("suite_name"),
                    "security": security,
                    "utility": trajectory.get("utility"),
                    "attack_success": (not security) if isinstance(security, bool) else None,
                    "gold_attack_success": (not security) if isinstance(security, bool) else None,
                    "attack_type": trajectory.get("attack_type"),
                    "user_task_id": trajectory.get("user_task_id"),
                    "injection_task_id": trajectory.get("injection_task_id"),
                    "polluted_tool_message_index": selected_case.polluted_tool_index,
                    "target_assistant_message_index": selected_case.target_assistant_index,
                    "target_message": trajectory["messages"][selected_case.target_assistant_index],
                    "player_text": player_texts(selected_case),
                })
                fact_chars, attack_chars = fact_attack_lengths(selected_case)
                record["fact_chars"] = fact_chars
                record["attack_chars"] = attack_chars
                if not isinstance(record["gold_attack_success"], bool):
                    raise ValueError("Missing boolean AgentDojo security label")
                prompts, target, kind = agentdojo_variants(model, selected_case)
                rendered = render_selected_target(model.tokenizer, selected_case)
                record["target_text"] = rendered.target_text
            record["target_kind"] = kind
            record["target_token_count"] = len(target)
            record["prompt_token_counts"] = [len(prompt_ids) for prompt_ids in prompts]
            values = []
            strategies = []
            for prompt_ids in prompts:
                value, strategy = score_fixed_action(
                    model, prompt_ids, target,
                    full_forward_token_threshold=args.full_forward_token_threshold,
                    streaming_chunk_size=args.streaming_chunk_size,
                    prefill_chunk_size=args.prefill_chunk_size,
                )
                values.append(value)
                strategies.append(strategy)
            record.update(loo_scores(*values, args.tau))
            record["scoring_strategies"] = strategies
            record["valid_for_stats"] = True
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
        record["loo_time_seconds"] = time.perf_counter() - started
        output.append(record)
        write_jsonl(output_path, output)

    write_jsonl(output_path, output)
    summary = summarize(
        output, dataset=args.dataset, model_config=args.model_config,
        corpus_audit=corpus_audit, selection=selection,
        tau=args.tau, tau_grid=args.tau_grid,
        comparison_audit=comparison_audit,
    )
    (output_dir / "results.causalarmor_loo.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
