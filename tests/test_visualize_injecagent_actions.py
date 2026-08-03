import importlib.util
import pathlib
import sys
import unittest


SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "visualize_injecagent_actions.py"
SPEC = importlib.util.spec_from_file_location("visualize_injecagent_actions", SCRIPT_PATH)
viz = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = viz
SPEC.loader.exec_module(viz)


class VisualizeInjecAgentActionsTest(unittest.TestCase):
    def _row(self):
        return {
            "case_id": 7,
            "eval": "succ",
            "attack_type": "Financial Harm",
            "attacker_tools": ["BankManagerPayBill"],
            "user_tool": "TwitterManagerReadTweet",
            "action_kind": "tool_action",
            "action_text": "Action: BankManagerPayBill\nAction Input: {'amount': 500}",
            "action_tool": "BankManagerPayBill",
            "valid_for_stats": True,
            "values": {
                "empty": -5.0,
                "auth": -4.0,
                "data_fact": -4.5,
                "data_attack": -2.0,
                "auth+data_fact+data_attack": -1.0,
            },
            "players": ["auth", "data_fact", "data_attack"],
            "phi_auth": 0.5,
            "phi_data_fact": 0.25,
            "phi_data_attack": 2.0,
            "shapley_attack_dominant": True,
        }

    def test_attack_margin(self):
        self.assertAlmostEqual(viz.attack_margin(self._row()), 1.25)

    def test_merge_attention_by_case_id(self):
        row = viz.merge_attention(
            self._row(),
            {
                "7": {
                    "case_id": 7,
                    "attention_shift": True,
                    "attention_shift_attack": True,
                    "attention_shift_basis": "auth_focus_score<=threshold_excluding_special",
                    "auth_focus_score": 0.0909,
                    "threshold": 0.5,
                    "attention_attack_dominant": True,
                    "region_scores": {"auth": 0.1, "data_fact": 0.2, "data_attack": 0.8, "special": 8.9},
                    "region_scores_normalized": {"auth": 0.01, "data_fact": 0.02, "data_attack": 0.08, "special": 0.89},
                    "region_scores_player_normalized": {
                        "auth": 0.0909,
                        "data_fact": 0.1818,
                        "data_attack": 0.7273,
                    },
                    "player_attention_mass": 1.1,
                    "top_tokens": [{"i": 1, "t": "Bank", "s": 0.8, "r": "data_attack"}],
                }
            },
        )
        self.assertTrue(row["attention_shift"])
        self.assertTrue(row["attention_shift_attack"])
        self.assertTrue(row["attention_attack_dominant"])
        self.assertIn("attention", row)

    def test_render_html_contains_core_sections(self):
        row = viz.merge_attention(
            self._row(),
            {
                "7": {
                    "case_id": 7,
                    "attention_shift": True,
                    "attention_shift_attack": True,
                    "attention_shift_basis": "auth_focus_score<=threshold_excluding_special",
                    "auth_focus_score": 0.0909,
                    "threshold": 0.5,
                    "attention_attack_dominant": True,
                    "region_scores": {"auth": 0.1, "data_fact": 0.2, "data_attack": 0.8, "special": 8.9},
                    "region_scores_normalized": {"auth": 0.01, "data_fact": 0.02, "data_attack": 0.08, "special": 0.89},
                    "region_scores_player_normalized": {
                        "auth": 0.0909,
                        "data_fact": 0.1818,
                        "data_attack": 0.7273,
                    },
                    "player_attention_mass": 1.1,
                    "top_tokens": [{"i": 1, "t": "Bank", "s": 0.8, "r": "data_attack"}],
                }
            },
        )
        html = viz.render_html([row], {"shapley": "s.jsonl", "attention": "a.jsonl"}, "Demo")
        self.assertIn("Explained Action", html)
        self.assertNotIn("Coalition Values", html)
        self.assertIn("Tokens", html)
        self.assertIn('data-filter="auth"', html)
        self.assertNotIn("ATTACK MARGIN", html)
        self.assertNotIn("Label/Action Mismatch", html)
        self.assertNotIn("Prompt Mass", html)
        self.assertNotIn("Non-Prompt Mass", html)
        self.assertNotIn("Player Mass", html)
        self.assertNotIn("Prompt-Normalized Sum", html)
        self.assertIn("Auth Focus", html)
        self.assertIn("Threshold", html)
        self.assertIn("Attack Dominant", html)
        self.assertIn("Player-Normalized Attention", html)
        self.assertIn("Prompt-Normalized Attention", html)
        self.assertIn('<details class="attention-details">', html)
        self.assertNotIn('<details class="attention-details" open>', html)
        self.assertIn("player norm 0.7273", html)
        self.assertIn("prompt norm 0.89", html)
        self.assertIn('<details class="record" id="case-7" data-eval="succ" data-action="tool_action" data-shapley="true" open>', html)

    def test_merge_judge_distinguishes_joint_method_and_status(self):
        row = {**self._row(), "target_scope": "full_action", "shapley_time_seconds": 1.25,
               "attention_time_seconds": 0.75}
        merged = viz.merge_judge(row, {("7", "full_action"): {
            "case_id": 7, "target_scope": "full_action",
            "judge": {"status": "judged", "behavior_label": "executed"},
            "attack_action_executed": True, "attack_success_strict": True,
        }})
        page = viz.render_html([merged], {"shapley": "s", "attention": "a", "judge": "j"}, "Demo")
        self.assertTrue(merged["attack_success_strict"])
        self.assertIn("Experiment Joint Method", page)
        self.assertIn("Judge Behavior", page)
        self.assertIn("Shapley Seconds", page)
        self.assertIn("Attention Seconds", page)

    def test_missing_judge_is_not_a_behavior_negative(self):
        merged = viz.merge_judge(self._row(), {})
        self.assertEqual(merged["judge"]["status"], "judge_unavailable")
        self.assertIsNone(merged["attack_action_executed"])
        self.assertFalse(merged["attack_success_strict"])

    def test_html_contains_eval_action_and_shapley_filters(self):
        html = viz.render_html([self._row()], {"shapley": "s"}, "Demo")
        self.assertIn('id="eval-filter"', html)
        self.assertIn('id="action-filter"', html)
        self.assertIn('id="shapley-filter"', html)
        self.assertIn("applyFilters", html)

    def test_unsuccessful_non_shift_case_is_collapsed(self):
        row = dict(self._row())
        row["eval"] = "unsucc"
        row["attention_shift"] = False
        row["attention_shift_attack"] = False
        html = viz.render_html([row], {"shapley": "s.jsonl", "attention": "a.jsonl"}, "Demo")
        self.assertIn('<details class="record" id="case-7" data-eval="unsucc" data-action="tool_action" data-shapley="true">', html)
        self.assertNotIn('<details class="record" id="case-7" data-eval="unsucc" data-action="tool_action" data-shapley="true" open>', html)


if __name__ == "__main__":
    unittest.main()
