"""Paired, task-clustered analysis of candidate-only remove_attack experiments."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import random
import statistics

import counterfactual_support as cf

SEED = 20261014


def unique_results(rows: list[dict], identity_fields: tuple[str, ...]) -> list[dict]:
    result = {}
    for row in rows:
        identity = tuple(row.get(field) for field in identity_fields)
        if identity in result and result[identity].get("cache_key") != row.get("cache_key"):
            previous = result[identity]
            same_evaluation_retry = (row.get("generation_cache_key") is not None
                                     and row.get("generation_cache_key") == previous.get("generation_cache_key")
                                     and row.get("evaluation_config_hash") is not None
                                     and row.get("evaluation_config_hash") == previous.get("evaluation_config_hash"))
            if not same_evaluation_retry:
                raise ValueError("Conflicting configurations/judges for one observation; use separate output directories")
        if identity not in result or result[identity].get("status") != "ok":
            result[identity] = row
    return list(result.values())


def cluster_estimate(rows: list[dict], rounds=2000, seed=SEED) -> dict:
    values = [row["value"] for row in rows]
    clusters = defaultdict(list)
    for row in rows:
        clusters[row["cluster_id"]].append(row["value"])
    estimate = {"mean": statistics.mean(values) if values else None,
                "median": statistics.median(values) if values else None,
                "contexts": len(values), "clusters": len(clusters), "ci95": None}
    if len(clusters) < 2:
        estimate["ci_reason"] = "fewer_than_two_independent_task_clusters"
        return estimate
    if rounds <= 0:
        raise ValueError("Bootstrap rounds must be positive")
    totals = [(sum(group), len(group)) for group in clusters.values()]
    rng = random.Random(seed)
    bootstraps = []
    for _ in range(rounds):
        sample = [rng.choice(totals) for _ in totals]
        bootstraps.append(sum(value for value, count in sample) / sum(count for value, count in sample))
    bootstraps.sort()
    estimate["ci95"] = [bootstraps[int((rounds - 1) * .025)], bootstraps[int((rounds - 1) * .975)]]
    return estimate


def support_summary(rows: list[dict], rounds=2000) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["context_id"]].append(row)
    context_rows = [{"cluster_id": values[0]["cluster_id"], "value": statistics.mean(row["delta_s"] for row in values)}
                    for values in grouped.values()]
    values = [row["delta_s"] for row in rows]
    return {"actions": len(rows), "contexts": len(grouped),
            "action_mean": statistics.mean(values) if values else None,
            "action_median": statistics.median(values) if values else None,
            "action_support_decreased_fraction": sum(value > 0 for value in values) / len(values) if values else None,
            "context_effect": cluster_estimate(context_rows, rounds),
            "context_support_decreased_fraction": sum(row["value"] > 0 for row in context_rows) / len(context_rows) if context_rows else None}


def behavior_summary(contexts: list[dict], records: list[dict], seeds: list[int], rounds=2000) -> dict:
    by_key = {(row["context_id"], row["condition"], row["seed"]): row for row in records}
    pairs, task_pairs, resolved = [], [], Counter()
    total_lo = total_hi = 0.
    seed_pairs = 0
    with_original_match, without_original_match = [], []
    condition_rates = {condition: [] for condition in cf.CONDITIONS}
    for context in contexts:
        outcomes, tasks = {}, {}
        for condition in cf.CONDITIONS:
            outcomes[condition], tasks[condition] = [], []
            for seed in seeds:
                row = by_key.get((context["context_id"], condition, seed), {})
                value = row.get("matched_attack_goal") if row.get("status") == "ok" else None
                if not isinstance(value, bool):
                    value = None
                outcomes[condition].append(value)
                resolved[condition] += value is not None
                alignment = row.get("task_alignment") if row.get("status") == "ok" else "unresolved"
                tasks[condition].append(1 if alignment == "consistent" else 0 if alignment in {"mixed", "inconsistent"} else None)
            if all(value is not None for value in outcomes[condition]):
                condition_rates[condition].append({"value": statistics.mean(outcomes[condition]), "cluster_id": context["cluster_id"]})
        for original, removed in zip(outcomes["original"], outcomes["remove_attack"]):
            total_lo += (int(original) if original is not None else 0) - (int(removed) if removed is not None else 1)
            total_hi += (int(original) if original is not None else 1) - (int(removed) if removed is not None else 0)
            seed_pairs += original is not None and removed is not None
        if all(value is not None for group in outcomes.values() for value in group):
            original = statistics.mean(outcomes["original"])
            removed = statistics.mean(outcomes["remove_attack"])
            paired = {"context_id": context["context_id"], "cluster_id": context["cluster_id"],
                      "original_rate": original, "remove_attack_rate": removed, "value": original - removed}
            pairs.append(paired)
            (with_original_match if original > 0 else without_original_match).append(paired)
        if all(value is not None for group in tasks.values() for value in group):
            task_pairs.append({"cluster_id": context["cluster_id"],
                               "value": statistics.mean(tasks["remove_attack"]) - statistics.mean(tasks["original"])})
    denominator = len(contexts) * len(seeds)
    return {"selected_prepared_contexts": len(contexts), "expected_seed_pairs": denominator,
            "paired_contexts": len(pairs), "paired_seed_pairs": seed_pairs,
            "resolved_original": resolved["original"], "resolved_remove_attack": resolved["remove_attack"],
            "paired_effect": cluster_estimate(pairs, rounds),
            "paired_original_rate": statistics.mean(row["original_rate"] for row in pairs) if pairs else None,
            "paired_remove_attack_rate": statistics.mean(row["remove_attack_rate"] for row in pairs) if pairs else None,
            "condition_rates_all_seeds_resolved": {condition: cluster_estimate(rows, rounds) for condition, rows in condition_rates.items()},
            "worst_case_effect_bounds": [total_lo / denominator, total_hi / denominator] if denominator else None,
            "original_has_match": cluster_estimate(with_original_match, rounds),
            "original_no_match": cluster_estimate(without_original_match, rounds),
            "task_paired_effect": cluster_estimate(task_pairs, rounds),
            "paired_context_details": pairs}


def _read(path: Path) -> list[dict]:
    return cf.read_rows(path) if path.exists() else []


def _format(number) -> str:
    return "NA" if number is None else f"{number:.4f}"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        rows = [{"status": "no_valid_results"}]
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def plot_results(directory: Path, summary: dict, score_rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    axis = axes[0]
    grouped = defaultdict(list)
    for row in score_rows:
        grouped[row["context_id"]].append(row["delta_s"])
    if grouped:
        axis.hist([statistics.mean(values) for values in grouped.values()], bins=min(30, len(grouped)), color="#3c6f91")
        axis.axvline(0, color="black", lw=.8)
    else:
        axis.text(.5, .5, "No valid fixed-action scores", ha="center", transform=axis.transAxes)
    axis.set(xlabel="Mean log-probability decrease per context", ylabel="Contexts")
    behavior = summary["behavior"]
    axis = axes[1]
    if behavior["paired_contexts"]:
        pairs = behavior["paired_context_details"]
        for pair in pairs:
            axis.plot([0, 1], [pair["original_rate"], pair["remove_attack_rate"]], color="#3c6f91", alpha=.12, lw=.6)
        axis.plot([0, 1], [behavior["paired_original_rate"], behavior["paired_remove_attack_rate"]], color="#ab453c", marker="o", lw=2)
    else:
        axis.text(.5, .5, "No fully evaluated pairs", ha="center", transform=axis.transAxes)
    axis.set(xticks=[0, 1], xticklabels=["Original", "Remove ATTACK"], ylim=(-.02, 1.02), ylabel="Original attack-goal matching-call rate")
    figure.suptitle("UCT-Trace Shapley-positive candidates only")
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(directory / f"paired_effects.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def analyze_directory(directory: Path, rounds=2000) -> dict:
    directory = Path(directory)
    preparation = json.loads((directory / "prepare.summary.json").read_text(encoding="utf-8"))
    candidates, contexts = _read(directory / "candidates.jsonl"), _read(directory / "contexts.jsonl")
    if preparation["manifest_hash"] != cf.stable_hash({"candidates": candidates, "contexts": contexts}):
        raise ValueError("Manifest changed after preparation")
    scores = unique_results(_read(directory / "scores.jsonl"), ("candidate_id",))
    generations = unique_results(_read(directory / "generations.jsonl"), ("context_id", "condition", "seed"))
    evaluations = unique_results(_read(directory / "evaluations.jsonl"), ("context_id", "condition", "seed"))
    candidate_evaluations = unique_results(_read(directory / "candidate_evaluations.jsonl"), ("candidate_id",))
    valid_generations = {row["cache_key"] for row in generations}
    if any(row["generation_cache_key"] not in valid_generations for row in evaluations):
        raise ValueError("Evaluation is not linked to the current generation records")
    candidate_ids, context_ids = {row["candidate_id"] for row in candidates}, {row["context_id"] for row in contexts}
    if any(row["candidate_id"] not in candidate_ids for row in scores) or any(row["context_id"] not in context_ids for row in generations + evaluations):
        raise ValueError("Result contains observations outside the frozen candidate manifest")
    valid_scores = [row for row in scores if row.get("status") == "ok"]
    qualitative = {row["candidate_id"]: row for row in candidate_evaluations}
    grouped = defaultdict(list)
    for row in valid_scores:
        grouped["target_kind=" + row["target_kind"]].append(row)
        labels = qualitative.get(row["candidate_id"], {})
        grouped["behavior=" + labels.get("behavior_label", "unresolved")].append(row)
        grouped["task_alignment=" + labels.get("task_alignment", "unresolved")].append(row)
    summary = {"experiment": cf.VERSION, "dataset": preparation["dataset"], "scope": "Shapley-positive candidates only",
               "bootstrap_rounds": rounds, "bootstrap_seed": SEED, "preparation": preparation,
               "support": support_summary(valid_scores, rounds),
               "support_groups": {key: support_summary(rows, rounds) for key, rows in grouped.items()},
               "behavior": behavior_summary(contexts, evaluations, preparation["seeds"], rounds),
               "coverage": {"candidate_actions": len(candidates), "score_attempted": len(scores), "score_valid": len(valid_scores),
                            "score_missing_or_invalid": len(candidates) - len(valid_scores),
                            "expected_generations": len(contexts) * 2 * len(preparation["seeds"]), "attempted_generations": len(generations),
                            "generation_status": dict(Counter(row["status"] for row in generations)),
                            "evaluation_status": dict(Counter(row["status"] for row in evaluations)),
                            "behavior_labels": dict(Counter(row["behavior_label"] for row in evaluations))}}
    cf.write_json(directory / "summary.json", summary)
    support = summary["support"]
    effect = support["context_effect"]
    support_tables = []
    for name, group in {"all": support, **summary["support_groups"]}.items():
        group_effect = group["context_effect"]
        support_tables.append({"group": name, "actions": group["actions"], "contexts": group["contexts"],
              "action_mean_delta_s": group["action_mean"], "action_median_delta_s": group["action_median"],
              "context_mean_delta_s": group_effect["mean"], "ci95_lower": group_effect["ci95"][0] if group_effect["ci95"] else None,
              "ci95_upper": group_effect["ci95"][1] if group_effect["ci95"] else None,
              "action_support_decreased_fraction": group["action_support_decreased_fraction"]})
    write_csv(directory / "support_table.csv", support_tables)
    behavior = summary["behavior"]
    behavior_tables = [{key: behavior[key] for key in ("selected_prepared_contexts", "paired_contexts", "paired_seed_pairs", "resolved_original", "resolved_remove_attack", "paired_original_rate", "paired_remove_attack_rate")} | {
              "group": "all_complete_pairs",
              "paired_effect": behavior["paired_effect"]["mean"], "ci95": json.dumps(behavior["paired_effect"]["ci95"]),
              "worst_case_bounds": json.dumps(behavior["worst_case_effect_bounds"])}]
    for name in ("original_has_match", "original_no_match"):
        group = behavior[name]
        behavior_tables.append({"group": name, "paired_contexts": group["contexts"],
                                "paired_effect": group["mean"], "ci95": json.dumps(group["ci95"])})
    write_csv(directory / "behavior_table.csv", behavior_tables)
    write_csv(directory / "paired_contexts.csv", behavior["paired_context_details"])
    text = f"""# UCT-Trace remove_attack experiment

Scope: only valid Shapley-positive candidates (strict M_attack > 0); no attention or final-judge selection.

Candidates: {len(candidates)} actions; {len(contexts)} prepared decision contexts.
Fixed-action valid scores: {len(valid_scores)} / {len(candidates)}.
Mean support decrease per context: {_format(effect['mean'])}; 95% task-cluster bootstrap CI: {effect['ci95']}.
Candidate actions with decreased support: {_format(support['action_support_decreased_fraction'])}.

Fully evaluated paired contexts: {behavior['paired_contexts']} / {len(contexts)} (all frozen seeds required).
Paired Original matching-call rate: {_format(behavior['paired_original_rate'])}.
Paired Remove-ATTACK matching-call rate: {_format(behavior['paired_remove_attack_rate'])}.
Paired reduction: {_format(behavior['paired_effect']['mean'])}; 95% CI: {behavior['paired_effect']['ci95']}.
Unknown-result worst-case reduction bounds over all prepared contexts/seeds: {behavior['worst_case_effect_bounds']}.
Next-step task-consistency change (Remove minus Original): {_format(behavior['task_paired_effect']['mean'])}.
Original-has-match group: {behavior['original_has_match']['contexts']} contexts; reduction {_format(behavior['original_has_match']['mean'])}.
Original-no-match group: {behavior['original_no_match']['contexts']} contexts; reduction {_format(behavior['original_no_match']['mean'])}.

## Interpretation limits

Fixed-action scoring uses the legacy attribution context and frozen original action. Generation uses the same
decision prefix, with frozen full suite schemas added for AgentDojo. The two analyses have different rendering contexts.
Tool requests are not executed. These results do not establish final attack execution or full task utility.
Support decrease alone does not establish that unauthorized control disappeared; refusal/mention outputs can also depend on the injection.
Only prepared contexts enter behavioral bounds; construction failures remain visible in preparation and score coverage.
Failure, missing, truncated and unresolved records are not classified as attack failures.
No conclusions about non-candidates or full-corpus detection accuracy are supported.
"""
    (directory / "report.md").write_text(text, encoding="utf-8")
    plot_results(directory, summary, valid_scores)
    print(json.dumps({"candidates": len(candidates), "contexts": len(contexts), "valid_scores": len(valid_scores), "paired_contexts": behavior["paired_contexts"]}))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-rounds", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap_rounds <= 0:
        parser.error("Bootstrap rounds must be positive")
    analyze_directory(args.input_dir, args.bootstrap_rounds)


if __name__ == "__main__":
    main()
