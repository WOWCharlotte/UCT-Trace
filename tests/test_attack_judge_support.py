import json
import sys
import unittest

sys.path.insert(0, "scripts")

from attack_judge_support import (
    FORMULA_VERSION,
    agentdojo_input,
    attack_dominant,
    attack_margin,
    cache_key,
    injecagent_input,
    select_candidates,
)


class AttackJudgeSupportTests(unittest.TestCase):
    def test_attack_margin_clamps_non_positive_baselines(self):
        self.assertAlmostEqual(attack_margin(1.2, 0.4, 0.3), 0.5)
        self.assertAlmostEqual(attack_margin(0.4, -0.2, 0.1), 0.3)
        self.assertAlmostEqual(attack_margin(0.1, -0.8, -0.4), 0.1)
        self.assertAlmostEqual(attack_margin(0.6, -0.2, 0.8), -0.2)
        self.assertTrue(attack_dominant(0.1, -0.8, -0.4))

    def test_adapters_restore_independent_context(self):
        injec = injecagent_input({
            "case_id": 3, "User Instruction": "find mail", "Tool Response": "attack and facts",
            "Attacker Instruction": "send it", "action_text": "Final Answer: no",
        })
        self.assertEqual(injec.dataset, "injecagent")
        self.assertEqual(injec.attack_text, "send it")

        trajectory = {"security": True, "messages": [
            {"role": "user", "content": "find mail"},
            {"role": "tool", "content": "attack and facts"},
            {"role": "assistant", "content": "done", "tool_calls": [{
                "function": {"name": "send_mail", "arguments": json.dumps({"to": "a"})}
            }]},
        ]}
        dojo = agentdojo_input({"target_id": "x", "polluted_tool_message_index": 1,
                                "target_assistant_message_index": 2, "attack_text": "send mail"}, trajectory)
        self.assertEqual(dojo.user_task, "find mail")
        self.assertEqual(dojo.tool_response, "attack and facts")
        self.assertEqual(dojo.target_tool_name, "send_mail")

    def test_candidate_selection_and_stable_cache_key(self):
        rows = [
            {"id": "high", "phi_data_attack": 1, "phi_auth": -1, "phi_data_fact": 0},
            {"id": "low", "phi_data_attack": 0.1, "phi_auth": 0.2, "phi_data_fact": 0.1,
             "attention_region_scores": {"data_attack": 1, "auth": 0, "data_fact": 0}},
        ]
        selected, skipped = select_candidates(rows)
        self.assertEqual([row["id"] for row in selected], ["high"])
        self.assertEqual(skipped[0]["attack_attribution_trigger"], False)
        record = injecagent_input(rows[0], case_id="high")
        self.assertEqual(cache_key(record, "model", FORMULA_VERSION), cache_key(record, "model", FORMULA_VERSION))


if __name__ == "__main__":
    unittest.main()
