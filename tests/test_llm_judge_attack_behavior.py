import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, "scripts")

import llm_judge_attack_behavior as judge
from attack_judge_support import injecagent_input


class JudgeTests(unittest.TestCase):
    def test_parser_accepts_three_labels_and_rejects_invalid(self):
        for label in ("executed", "mentioned", "refused"):
            result = judge.parse_judge_response(json.dumps({
                "behavior_label": label, "attack_action_executed": label == "executed",
                "attack_intent_adopted": label != "refused", "evidence": "e", "confidence": .8,
            }))
            self.assertEqual(result["status"], "judged")
        self.assertEqual(judge.parse_judge_response("not json")["status"], "judge_failed")

    def test_cache_hit_does_not_request_again_and_strict_rule_is_separate(self):
        context = injecagent_input({"case_id": "x", "User Instruction": "u", "Tool Response": "r",
                                    "Attacker Instruction": "a", "action_text": "o"})
        record = {"judge_input": context.as_dict(), "attack_attribution_trigger": True}
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as handle:
            cache_path = handle.name
            json.dump({}, handle)
        response = {"status": "judged", "behavior_label": "executed", "attack_action_executed": True,
                    "attack_intent_adopted": True, "evidence": "tool", "confidence": 1}
        with patch.object(judge, "request_judge", return_value=response) as request:
            first = judge.judge_records([record], {"model": "m", "api_key": "k", "base_url": "u"}, cache_path)
            second = judge.judge_records([record], {"model": "m", "api_key": "k", "base_url": "u"}, cache_path)
        self.assertEqual(request.call_count, 1)
        self.assertTrue(first[0]["attack_success_strict"])
        self.assertTrue(second[0]["judge"]["cache_hit"])

    def test_concurrency_limit_is_enforced(self):
        with self.assertRaises(ValueError):
            judge.judge_records([], {"model": "m", "api_key": "k", "base_url": "u"}, max_concurrency=11)

    def test_low_contribution_is_not_sent_to_judge_or_labeled_false(self):
        context = injecagent_input({"case_id": "low", "User Instruction": "u", "Tool Response": "r",
                                    "Attacker Instruction": "a", "action_text": "o"})
        record = {"judge_input": context.as_dict(), "attack_attribution_trigger": False, "eval": "succ",
                  "security": False, "utility": 7}
        with patch.object(judge, "request_judge") as request:
            result = judge.judge_records([record], {"model": "m", "api_key": "k", "base_url": "u"})
        request.assert_not_called()
        self.assertEqual(result[0]["judge"]["status"], "not_judged_by_cascade")
        self.assertIsNone(result[0]["attack_action_executed"])
        self.assertFalse(result[0]["attack_success_strict"])
        self.assertEqual((result[0]["eval"], result[0]["security"], result[0]["utility"]), ("succ", False, 7))


if __name__ == "__main__":
    unittest.main()
