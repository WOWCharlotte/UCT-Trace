import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import analyze_counterfactual_experiment as analysis


class CounterfactualAnalysisTests(unittest.TestCase):
    def test_failed_and_missing_pairs_have_worst_case_bounds(self):
        contexts = [{"context_id": "a", "cluster_id": "a"}, {"context_id": "b", "cluster_id": "b"}]
        records = [{"context_id": "a", "condition": "original", "seed": 1, "matched_attack_goal": True,
                    "task_alignment": "inconsistent", "status": "ok"},
                   {"context_id": "a", "condition": "remove_attack", "seed": 1, "matched_attack_goal": False,
                    "task_alignment": "consistent", "status": "ok"},
                   {"context_id": "b", "condition": "original", "seed": 1, "matched_attack_goal": True,
                    "task_alignment": "unresolved", "status": "ok"},
                   {"context_id": "b", "condition": "remove_attack", "seed": 1, "matched_attack_goal": None,
                    "task_alignment": "unresolved", "status": "failed"}]
        result = analysis.behavior_summary(contexts, records, [1], rounds=20)
        self.assertEqual(result["paired_contexts"], 1)
        self.assertEqual(result["paired_effect"]["mean"], 1)
        self.assertEqual(result["worst_case_effect_bounds"], [.5, 1.])
        self.assertEqual(result["resolved_original"], 2)
        self.assertEqual(result["resolved_remove_attack"], 1)
        self.assertEqual(result["task_paired_effect"]["mean"], 1)

    def test_contexts_not_seed_rows_determine_effect_and_interval(self):
        contexts = [{"context_id": "a", "cluster_id": "shared"}, {"context_id": "b", "cluster_id": "shared"}]
        records = []
        for context in contexts:
            for seed in (1, 2):
                for condition in ("original", "remove_attack"):
                    records.append({"context_id": context["context_id"], "condition": condition, "seed": seed,
                                    "matched_attack_goal": condition == "original", "task_alignment": "unresolved", "status": "ok"})
        result = analysis.behavior_summary(contexts, records, [1, 2], rounds=20)
        self.assertEqual(result["paired_contexts"], 2)
        self.assertEqual(result["paired_seed_pairs"], 4)
        self.assertEqual(result["paired_effect"]["clusters"], 1)
        self.assertIsNone(result["paired_effect"]["ci95"])

    def test_support_averages_actions_within_context_before_bootstrap(self):
        rows = [{"context_id": "a", "cluster_id": "a", "delta_s": 1.},
                {"context_id": "a", "cluster_id": "a", "delta_s": 3.},
                {"context_id": "b", "cluster_id": "b", "delta_s": -2.}]
        summary = analysis.support_summary(rows, rounds=20)
        self.assertEqual(summary["contexts"], 2)
        self.assertEqual(summary["actions"], 3)
        self.assertEqual(summary["context_effect"]["mean"], 0.)
        self.assertAlmostEqual(summary["action_mean"], 2 / 3)

    def test_conflicting_configurations_are_not_averaged(self):
        rows = [{"context_id": "a", "condition": "original", "seed": 1, "cache_key": "x"},
                {"context_id": "a", "condition": "original", "seed": 1, "cache_key": "y"}]
        with self.assertRaises(ValueError):
            analysis.unique_results(rows, ("context_id", "condition", "seed"))

    def test_bootstrap_is_reproducible(self):
        rows = [{"cluster_id": str(i // 2), "value": float(i)} for i in range(8)]
        self.assertEqual(analysis.cluster_estimate(rows, rounds=100), analysis.cluster_estimate(rows, rounds=100))

    def test_evaluation_retry_after_generation_failure_preserves_observation(self):
        base = {"context_id": "a", "condition": "original", "seed": 1,
                "generation_cache_key": "same-input", "evaluation_config_hash": "same-judge"}
        rows = [{**base, "cache_key": "failed-generation", "status": "failed"},
                {**base, "cache_key": "retried-generation", "status": "ok"}]
        selected = analysis.unique_results(rows, ("context_id", "condition", "seed"))
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["status"], "ok")


if __name__ == "__main__":
    unittest.main()
