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
    clear_cuda_cache,
    embedding_layer,
    locate_source,
    mean_logprob_with_masked_regions,
    model_device,
    original_user_message,
    render_selected_target,
)
from baselines.attention_tracker.utils import create_model, open_config
from injecagent_action_experiment import (
    build_injecagent_prompt_parts,
    build_tool_index,
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


def injecagent_variants(model: Any, source: dict[str, Any], tools: dict[str, dict[str, Any]]) -> tuple[list[list[int]], list[int], str]:
    parts = build_injecagent_prompt_parts(source, tools)
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


def metrics(rows: list[dict[str, Any]], prediction_field: str) -> dict[str, Any]:
    tp = sum(row[prediction_field] and row["gold_attack_success"] for row in rows)
    fp = sum(row[prediction_field] and not row["gold_attack_success"] for row in rows)
    tn = sum(not row[prediction_field] and not row["gold_attack_success"] for row in rows)
    fn = sum(not row[prediction_field] and row["gold_attack_success"] for row in rows)
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


def read_manifest(path: str, dataset: str) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    key = "case_id" if dataset == "injecagent" else "target_id"
    selected = [row for row in rows if row.get("valid_for_stats") is True]
    if dataset == "injecagent":
        selected = [row for row in selected if row.get("target_scope") == "full_action"]
    seen: set[str] = set()
    for row in selected:
        identity = str(row.get(key, ""))
        if not identity or identity in seen:
            raise ValueError(f"Missing or duplicate {key} in Shapley manifest: {identity!r}")
        seen.add(identity)
    if not selected:
        raise ValueError("Shapley manifest has no valid fixed-action rows")
    return selected


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(rows: list[dict[str, Any]], *, dataset: str, model_config: str, manifests: list[str], tau: float) -> dict[str, Any]:
    valid = [row for row in rows if row.get("valid_for_stats") is True]
    tool_kinds = {"tool_action"} if dataset == "injecagent" else {"tool_calls", "mixed"}
    tool_rows = [row for row in valid if row["target_kind"] in tool_kinds]
    times = [row["loo_time_seconds"] for row in valid]
    def comparison(subset: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "loo": metrics(subset, "predicted_attack_success"),
            "shapley_only": metrics(subset, "shapley_attack_dominant"),
        }
    return {
        "schema_version": 1,
        "method": "causalarmor_tool_result_loo_offline_adaptation",
        "dataset": dataset,
        "model_config": model_config,
        "shapley_manifests": manifests,
        "tau": tau,
        "selected_count": len(rows),
        "valid_count": len(valid),
        "failed_count": len(rows) - len(valid),
        "overall": comparison(valid),
        "tool_action_subset": comparison(tool_rows),
        "runtime_seconds": {
            "mean": statistics.mean(times) if times else None,
            "median": statistics.median(times) if times else None,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("injecagent", "agentdojo"), required=True)
    parser.add_argument("--shapley-manifest", nargs="+", required=True, help="Existing UCT-Trace Shapley JSONL; pass success and failure files together for InjecAgent")
    parser.add_argument("--model-config", required=True, help="Existing UCT-Trace model configuration JSON")
    parser.add_argument("--output-dir", required=True, help="New directory; existing attribution files are untouched")
    parser.add_argument("--input", nargs="+", help="InjecAgent generated cases JSONL, in the same order as the Shapley manifests")
    parser.add_argument("--tools", default="data/injecagent/injecagent_data/tools.json")
    parser.add_argument("--input-root", help="AgentDojo trajectory root")
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--limit", type=int, help="Limit manifest rows for a later small run")
    parser.add_argument("--full-forward-token-threshold", type=int, default=1024)
    parser.add_argument("--streaming-chunk-size", type=int, default=16)
    parser.add_argument("--prefill-chunk-size", type=int, default=256)
    args = parser.parse_args()
    if args.dataset == "injecagent" and (not args.input or len(args.input) != len(args.shapley_manifest)):
        parser.error("InjecAgent requires one --input per --shapley-manifest, in matching order")
    if args.dataset == "agentdojo" and not args.input_root:
        parser.error("--input-root is required for AgentDojo")
    if args.dataset == "agentdojo" and len(args.shapley_manifest) != 1:
        parser.error("AgentDojo accepts exactly one --shapley-manifest per run")
    if not math.isfinite(args.tau):
        parser.error("--tau must be finite")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.streaming_chunk_size <= 0 or args.prefill_chunk_size <= 0:
        parser.error("chunk sizes must be positive")
    return args


def main() -> None:
    args = parse_args()
    manifest = [
        (group, manifest_path, row)
        for group, manifest_path in enumerate(args.shapley_manifest)
        for row in read_manifest(manifest_path, args.dataset)
    ]
    if args.limit is not None:
        manifest = manifest[:args.limit]
    if args.dataset == "injecagent":
        source_rows = [read_jsonl(path) for path in args.input]
        tool_index = build_tool_index(read_json(args.tools))
        cases = None
    else:
        selected, _, _ = audit_corpus(args.input_root)
        cases = {case.target_id: case for case in selected}
        source_rows = None
        tool_index = None
    model = create_model(open_config(args.model_config))
    output: list[dict[str, Any]] = []
    for group, manifest_path, existing in manifest:
        identity = existing["case_id"] if args.dataset == "injecagent" else existing["target_id"]
        record: dict[str, Any] = {
            "schema_version": 1,
            "dataset": args.dataset,
            "sample_id": f"{group}:{identity}" if args.dataset == "injecagent" else identity,
            "shapley_manifest": manifest_path,
            "input_source": args.input[group] if args.dataset == "injecagent" else args.input_root,
            "case_id": existing.get("case_id"),
            "target_id": existing.get("target_id"),
            "target_text": existing.get("target_text"),
            "target_kind": existing.get("action_kind") if args.dataset == "injecagent" else existing.get("target_kind"),
            "source_path": existing.get("source_path"),
            "gold_attack_success": existing.get("eval") == "succ" if args.dataset == "injecagent" else existing.get("attack_success"),
            "shapley_attack_dominant": existing.get("shapley_attack_dominant", existing.get("attack_dominant")),
            "valid_for_stats": False,
            "ablation_unit": "original_user_content_vs_entire_polluted_tool_result_content",
        }
        started = time.perf_counter()
        try:
            if args.dataset == "injecagent" and existing.get("eval") not in {"succ", "unsucc"}:
                raise ValueError("InjecAgent Shapley manifest has no success/failure label")
            if not isinstance(record["gold_attack_success"], bool):
                raise ValueError("Missing boolean attack-success label")
            if not isinstance(record["shapley_attack_dominant"], bool):
                raise ValueError("Missing Shapley-only comparison prediction")
            if args.dataset == "injecagent":
                source = source_rows[group][int(identity)]
                prompts, target, kind = injecagent_variants(model, source, tool_index)
                if parse_action(source.get("output", "")).text != existing.get("target_text"):
                    raise ValueError("Fixed InjecAgent action differs from Shapley manifest")
            else:
                selected_case = cases.get(identity)
                if selected_case is None:
                    raise ValueError("AgentDojo target_id not found in input corpus")
                prompts, target, kind = agentdojo_variants(model, selected_case)
                if model.tokenizer.decode(target, skip_special_tokens=False) != existing.get("target_text"):
                    raise ValueError("Fixed AgentDojo action differs from Shapley manifest")
            record["target_kind"] = kind
            record["target_token_count"] = len(target)
            record["prompt_token_counts"] = [len(prompt_ids) for prompt_ids in prompts]
            values = []
            strategies = []
            for prompt_ids in prompts:
                value, strategy = score_fixed_action(
                    model,
                    prompt_ids,
                    target,
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
    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "results.causalarmor_loo.jsonl", output)
    summary = summarize(output, dataset=args.dataset, model_config=args.model_config, manifests=args.shapley_manifest, tau=args.tau)
    (output_dir / "results.causalarmor_loo.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
