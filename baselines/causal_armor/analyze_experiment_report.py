"""Recompute the report's tables from local results; no inference or input edits."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
MODELS = {"qwen3": "Qwen3-8B", "phi3": "Phi-3-mini-128k", "llama3": "Llama3-8B"}
AD = {"qwen3": "agentdojo_fact_longer_than_attack_qwen_8b_chunk16_fixed", "phi3": "agentdojo_phi3_mini_128k_fixed", "llama3": "agentdojo_llama3_8b"}


def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def metrics(rows, field="predicted_attack_success", tau=None):
    c = Counter()
    for r in rows:
        p = r[field] if tau is None else r["dominance_margin"] > -tau
        y = r["gold_attack_success"]
        assert isinstance(p, bool) and isinstance(y, bool)
        c["tp" if p and y else "fp" if p else "fn" if y else "tn"] += 1
    tp, fp, tn, fn = [c[k] for k in ("tp", "fp", "tn", "fn")]
    div = lambda a, b: a / b if b else 0.0
    return dict(count=len(rows), tp=tp, fp=fp, tn=tn, fn=fn,
                accuracy=div(tp+tn, len(rows)), precision=div(tp, tp+fp),
                recall=div(tp, tp+fn), fpr=div(fp, fp+tn), f1=div(2*tp, 2*tp+fp+fn))


def percentile(values, q):
    x = sorted(values)
    i = (len(x)-1)*q
    lo = math.floor(i)
    return x[lo] + (x[min(lo+1, len(x)-1)]-x[lo])*(i-lo)


def main():
    results = []
    hashes = {}
    for dataset in ("injecagent", "agentdojo"):
        for model, label in MODELS.items():
            folder = ROOT / "result" / f"causalarmor_loo_{dataset}_{model}"
            source = folder / "results.causalarmor_loo.jsonl"
            summary_path = folder / "results.causalarmor_loo.summary.json"
            rows = read(source)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            valid = [r for r in rows if r.get("valid_for_stats") is True]
            assert len({r["sample_id"] for r in rows}) == len(rows)
            assert (len(rows), len(valid), len(rows)-len(valid)) == (summary["selected_count"], summary["valid_count"], summary["failed_count"])
            for r in valid:
                assert r["tau"] == 0
                du = r["logprob_full"]-r["logprob_without_user"]
                dt = r["logprob_full"]-r["logprob_without_tool_result"]
                for actual, expected in [(r["delta_user"], du), (r["delta_tool_result"], dt), (r["dominance_margin"], dt-du)]:
                    assert math.isfinite(actual) and math.isclose(actual, expected, abs_tol=1e-10)
                assert r["predicted_attack_success"] == (dt-du > 0)
            overall = metrics(valid)
            for k, v in overall.items():
                assert math.isclose(v, summary["overall"][k], abs_tol=1e-12)
            sweeps = [dict(tau=t, **metrics(valid, tau=t)) for t in (-1, -.4, 0, .4, 1)]
            for actual, expected in zip(sweeps, summary["tau_sweep"]):
                for k, v in actual.items():
                    assert math.isclose(v, expected[k], abs_tol=1e-12)
            if dataset == "agentdojo":
                paths = [ROOT / "result" / AD[model] / "results.shapley.jsonl"]
            else:
                date = "20260810" if model == "llama3" else "20260809"
                paths = [ROOT / "result" / f"injecagent_{model}_dh_{condition}_{date}" / "results.action_shapley.jsonl" for condition in ("attack", "clean")]
            uct = {}
            attempted = 0
            for group, path in enumerate(paths):
                ur = read(path)
                attempted += len(ur)
                for r in ur:
                    if r.get("valid_for_stats") is not True:
                        continue
                    if dataset == "injecagent" and r.get("target_scope") != "full_action":
                        continue
                    key = r["target_id"] if dataset == "agentdojo" else f'{group}:{r["case_id"]}'
                    assert key not in uct
                    uct[key] = r
            paired = []
            for r in valid:
                u = uct.get(r["sample_id"])
                if u is None:
                    continue
                assert r["target_text"] == u["target_text"], r["sample_id"]
                gold = u["attack_success"] if dataset == "agentdojo" else u["eval"] == "succ"
                assert gold == r["gold_attack_success"], (dataset, model, r["sample_id"], gold, r["gold_attack_success"])
                prediction = u.get("shapley_attack_dominant", u.get("attack_dominant"))
                assert prediction == (u["phi_data_attack"] > max(0, u["phi_auth"]) + max(0, u["phi_data_fact"]))
                paired.append(dict(r, uct_prediction=prediction))
            times = [r["loo_time_seconds"] for r in valid]
            runtime = dict(mean=statistics.mean(times), median=statistics.median(times), p95=percentile(times,.95), minimum=min(times), maximum=max(times), total=sum(times))
            for k in ("mean", "median"):
                assert math.isclose(runtime[k], summary["runtime_seconds"][k], abs_tol=1e-12)
            tool = [r for r in valid if r["target_kind"] in ("tool_action", "tool_calls", "mixed")]
            for k, v in metrics(tool).items():
                assert math.isclose(v, summary["tool_action_subset"][k], abs_tol=1e-12)
            groups = {}
            for field in ("suite_name", "target_kind", "attack_type"):
                groups[field] = {str(k): metrics([r for r in valid if r.get(field) == k]) for k in sorted({r.get(field) for r in valid}, key=str) if k is not None}
            margins = {}
            for gold in (True, False):
                rs = [r for r in valid if r["gold_attack_success"] == gold]
                margins[str(gold)] = {k: statistics.mean(r[k] for r in rs) for k in ("delta_user", "delta_tool_result", "dominance_margin")}
            results.append(dict(dataset=dataset, model=label, key=model, selected=len(rows), valid=len(valid), failed=len(rows)-len(valid), overall=overall,
                                uct_paths=[str(p.relative_to(ROOT)).replace('\\','/') for p in paths], uct_attempted=attempted, uct_valid=len(uct), overlap=len(paired),
                                paired_causalarmor=metrics(paired), paired_uct=metrics(paired,"uct_prediction"),
                                disagreements=dict(causalarmor_only=sum(r["predicted_attack_success"] and not r["uct_prediction"] for r in paired), uct_only=sum(not r["predicted_attack_success"] and r["uct_prediction"] for r in paired)),
                                runtime=runtime, tau_sweep=sweeps, tool_subset=metrics(tool), groups=groups, margins=margins,
                                failures=[{k:r.get(k) for k in ("sample_id","error","gold_attack_success")} for r in rows if r.get("valid_for_stats") is not True],
                                scoring_strategies=dict(Counter(s for r in valid for s in r["scoring_strategies"])) ))
            for p in [source, summary_path, *paths]:
                hashes[p.relative_to(ROOT).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    report = ROOT / "result/Attention_Shapley_综合实验报告_20260809.md"
    hashes[report.relative_to(ROOT).as_posix()] = hashlib.sha256(report.read_bytes()).hexdigest()
    configs = [ROOT / "configs/model_configs" / name for name in (
        "qwen3_8b-attn_config.json", "phi3-attn_config.json", "llama3_8b-attn_config.json")]
    for p in [OUT / "causalarmor_loo_baseline.py", OUT / "README.md", *configs]:
        hashes[p.relative_to(ROOT).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT / "experiment_audit_20261013.json").write_text(json.dumps(dict(results=results, sha256=hashes), ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    for r in results:
        print(r["dataset"], r["model"], "n=",r["overlap"], "CA",r["overall"], "UCT", r["paired_uct"])


if __name__ == "__main__":
    main()
