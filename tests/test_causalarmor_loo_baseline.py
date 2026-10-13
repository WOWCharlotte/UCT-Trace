import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest
from argparse import Namespace


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "baselines" / "causal_armor" / "causalarmor_loo_baseline.py"
SPEC = importlib.util.spec_from_file_location("causalarmor_loo_baseline", SCRIPT)
baseline = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = baseline
SPEC.loader.exec_module(baseline)


class CausalArmorScoresTest(unittest.TestCase):
    def test_loo_margin_and_strict_tau_rule(self):
        scores = baseline.loo_scores(-1.0, -1.2, -1.8, 0.0)
        self.assertAlmostEqual(scores["delta_user"], 0.2)
        self.assertAlmostEqual(scores["delta_tool_result"], 0.8)
        self.assertAlmostEqual(scores["dominance_margin"], 0.6)
        self.assertTrue(scores["predicted_attack_success"])
        tied = baseline.loo_scores(-1.0, -1.5, -1.5, 0.0)
        self.assertFalse(tied["predicted_attack_success"])

    def test_tau_sweep_reuses_margin(self):
        rows = [{
            "dominance_margin": -0.2,
            "predicted_attack_success": False,
            "gold_attack_success": True,
        }]
        self.assertEqual(baseline.metrics(rows, "predicted_attack_success", tau=0.0)["fn"], 1)
        self.assertEqual(baseline.metrics(rows, "predicted_attack_success", tau=0.4)["tp"], 1)

    def test_summary_keeps_attempts_failures_and_paired_metrics_separate(self):
        valid = {
            "valid_for_stats": True,
            "target_kind": "tool_action",
            "loo_time_seconds": 2.0,
            "predicted_attack_success": True,
            "shapley_attack_dominant": False,
            "gold_attack_success": True,
            "dominance_margin": 0.4,
        }
        summary = baseline.summarize(
            [valid, {"valid_for_stats": False}], dataset="injecagent",
            model_config="model.json", corpus_audit={"candidate_count": 2},
            selection={"source": "raw_injecagent_jsonl", "selected_count": 2},
            comparison_audit={"attempted_count": 3, "valid_count": 1},
            tau=0.0, tau_grid=[0.0, 0.4],
        )
        self.assertEqual(summary["selected_count"], 2)
        self.assertEqual(summary["valid_count"], 1)
        self.assertEqual(summary["failed_count"], 1)
        self.assertEqual(summary["posthoc_uct_trace_comparison"]["disagreements"]["causalarmor_only"], 1)
        self.assertEqual(len(summary["tau_sweep"]), 2)

    def test_raw_injecagent_selection_does_not_depend_on_comparison_manifest(self):
        rows = [
            {"User Instruction": "u1", "Tool Response": "t1", "Attacker Instruction": "a1", "output": "Final Answer: x", "eval": "unsucc"},
            {"User Instruction": "u2", "Tool Response": "t2", "Attacker Instruction": "a2", "output": "Final Answer: y", "eval": "succ"},
            {"User Instruction": "u3", "Tool Response": "t3", "Attacker Instruction": "a3", "output": "bad", "eval": "invalid"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "cases.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            args = Namespace(input=[str(path)], eval_filter="all", case_ids=None, limit=None)
            selected, audit = baseline.select_injecagent_rows(args)
        self.assertEqual([row["case_id"] for row in selected], [0, 1, 2])
        self.assertEqual(audit["selected_count"], 3)
        self.assertEqual(audit["source"], "raw_injecagent_jsonl")

    def test_optional_comparison_join_only_uses_valid_full_actions(self):
        rows = [
            {"case_id": 0, "valid_for_stats": True, "target_scope": "full_action", "attack_dominant": True},
            {"case_id": 1, "valid_for_stats": False, "target_scope": "full_action"},
            {"case_id": 2, "valid_for_stats": True, "target_scope": "tool_name"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "comparison.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            joined, audit = baseline.read_comparison_rows([str(path)], "injecagent")
        self.assertEqual(set(joined), {"0:0"})
        self.assertEqual(audit["attempted_count"], 3)
        self.assertEqual(audit["valid_count"], 1)

    def test_summary_without_comparison_is_standalone(self):
        row = {
            "valid_for_stats": True, "target_kind": "tool_action", "loo_time_seconds": 1.0,
            "predicted_attack_success": False, "gold_attack_success": False, "dominance_margin": -0.5,
        }
        summary = baseline.summarize(
            [row], dataset="injecagent", model_config="m.json",
            corpus_audit={"candidate_count": 1}, selection={"selected_count": 1},
            tau=0.0, tau_grid=[0.0],
        )
        self.assertNotIn("posthoc_uct_trace_comparison", summary)
        self.assertEqual(summary["overall"]["tn"], 1)


if __name__ == "__main__":
    unittest.main()
