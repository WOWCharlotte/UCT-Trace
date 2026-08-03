import importlib.util
import pathlib
import sys
import unittest


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "visualize_agentdojo_attribution.py"
SPEC = importlib.util.spec_from_file_location("visualize_agentdojo_attribution", SCRIPT)
viz = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = viz
SPEC.loader.exec_module(viz)


def base_row():
    return {
        "target_id": "banking/task/injection@tool-3:assistant-4",
        "case_id": "banking/task/important_instructions/injection",
        "suite_name": "banking",
        "security": False,
        "utility": True,
        "attack_success": True,
        "target_scope": "next_assistant_message",
        "target_kind": "tool_calls",
        "polluted_tool_message_index": 3,
        "target_assistant_message_index": 4,
        "target_text": "<script>send_money()</script>",
        "player_text": {"auth": "Pay bill", "data_fact": ["Bill: 10"], "data_attack": ["<INFORMATION>attack</INFORMATION>"]},
        "valid_for_stats": True,
    }


class AgentDojoVisualizationTest(unittest.TestCase):
    def test_merge_uses_target_id_and_keeps_both_methods(self):
        shapley = {**base_row(), "phi_auth": 0.1, "phi_data_fact": 0.2, "phi_data_attack": 0.3}
        attention = {**base_row(), "auth_focus_score": 0.25}
        merged = viz.merge_rows([shapley], [attention])
        self.assertEqual(len(merged), 1)
        self.assertIn("shapley", merged[0])
        self.assertIn("attention", merged[0])

    def test_merge_judge_keeps_ground_truth_separate(self):
        shapley = {**base_row(), "phi_auth": 0.1, "phi_data_fact": 0.2, "phi_data_attack": 0.3,
                   "shapley_time_seconds": 2.0}
        judged = {"target_id": base_row()["target_id"], "judge": {"status": "judged", "behavior_label": "mentioned"},
                  "attack_action_executed": False, "attack_success_strict": False}
        merged = viz.merge_rows([shapley], [], [judged])[0]
        self.assertFalse(merged["attack_success_strict"])
        self.assertEqual(merged["shapley"]["attack_success"], True)
        self.assertEqual(merged["judge"]["behavior_label"], "mentioned")

    def test_missing_judge_and_runtime_fields_are_rendered(self):
        row = {**base_row(), "shapley_time_seconds": 2.0, "attention_time_seconds": None}
        merged = viz.merge_rows([row], [])[0]
        page = viz.render_html([merged], {"shapley": "s"}, "Demo")
        self.assertIn("Ground-truth attack success", page)
        self.assertIn("Experiment joint method", page)
        self.assertIn("Shapley seconds", page)
        self.assertIn("judge_unavailable", page)

    def test_duplicate_target_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate target_id"):
            viz.row_index([base_row(), base_row()])

    def test_selects_single_target_or_all_with_limit(self):
        first = base_row()
        second = {**base_row(), "target_id": "second"}
        self.assertEqual(viz.select_rows([first, second], False, 0, "second", None)[0]["target_id"], "second")
        self.assertEqual(len(viz.select_rows([first, second], True, 0, None, 1)), 1)

    def test_html_has_agentdojo_controls_metrics_and_escaped_target(self):
        shapley = {
            **base_row(), "players": ["auth", "data_fact", "data_attack"],
            "phi_auth": 0.1, "phi_data_fact": -0.2, "phi_data_attack": 0.5,
            "shapley_attack_dominant": True, "efficiency_error": 0.0,
        }
        attention = {
            **base_row(), "region_scores_raw": {"auth": .1, "data_fact": .2, "data_attack": .3, "special": .4},
            "region_scores_prompt_normalized": {"auth": .1, "data_fact": .2, "data_attack": .3, "special": .4},
            "region_scores_player_normalized": {"auth": 1/6, "data_fact": 2/6, "data_attack": 3/6},
            "auth_focus_score": 1/6, "threshold": .5, "attention_shift": True,
            "attention_attack_dominant": False,
            "tokens": [{"i": 0, "t": "<bad>", "s": .1, "r": "special"}],
        }
        page = viz.render_html(viz.merge_rows([shapley], [attention]), {"shapley": "s.jsonl", "attention": "a.jsonl"}, "Demo")
        self.assertIn("Next Assistant Target", page)
        self.assertIn("Prompt-Normalized Attention Including SPECIAL", page)
        self.assertIn('id="suite-filter"', page)
        self.assertIn('data-region-toggle="data_attack"', page)
        self.assertIn("Polluted tool index", page)
        self.assertNotIn("<script>send_money()</script>", page)
        self.assertIn("&lt;script&gt;send_money()&lt;/script&gt;", page)


if __name__ == "__main__":
    unittest.main()
