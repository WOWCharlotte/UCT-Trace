import json
import math
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import counterfactual_support as cf


class CounterfactualSupportTests(unittest.TestCase):
    def test_candidates_require_valid_finite_strict_positive_margin(self):
        base = {"valid_for_stats": True, "phi_auth": -1, "phi_data_fact": .2,
                "phi_data_attack": .3, "case_id": 1, "action_text": "Final Answer: no"}
        rows = [base, {**base, "phi_data_attack": .2}, {**base, "phi_auth": None},
                {**base, "phi_data_attack": math.inf}, {**base, "valid_for_stats": False},
                {**base, "phi_data_attack": True}, {**base, "phi_auth": "NaN"}]
        selected, audit = cf.select_candidates(rows)
        self.assertEqual(len(selected), 1)
        self.assertAlmostEqual(selected[0]["source_margin"], .1)
        self.assertEqual(len(audit), 7)
        self.assertNotIn("source_margin", base)

    def test_deletion_multiple_blocks_preserves_every_fact_byte(self):
        text = '{"x":"fact <INFORMATION>one</INFORMATION> tail <INFORMATION>two</INFORMATION>"}'
        spans = cf.information_spans(text)
        removed, audit = cf.remove_spans(text, spans)
        self.assertEqual(removed, '{"x":"fact  tail "}')
        self.assertEqual(audit["removed_text"], ["<INFORMATION>one</INFORMATION>",
                                                "<INFORMATION>two</INFORMATION>"])
        with self.assertRaises(ValueError):
            cf.remove_spans(text, [(1, 9), (8, 10)])
        with self.assertRaises(ValueError):
            cf.remove_spans('{"x":1}', [(5, 6)])
        with self.assertRaises(ValueError):
            cf.remove_spans("facts", [])
        with self.assertRaises(ValueError):
            cf.information_spans("<INFORMATION>complete</INFORMATION><INFORMATION>open")

    def test_exact_attack_matches_do_not_delete_other_context(self):
        self.assertEqual(cf.literal_spans("attack fact attack", "attack"), [(0, 6), (12, 18)])
        with self.assertRaises(ValueError):
            cf.literal_spans("only facts", "attack")

    def test_parse_calls_and_schema_validation(self):
        tools = [{"name": "send", "parameters": {"type": "object",
                 "properties": {"to": {"type": "string"}}, "required": ["to"]}}]
        parsed = cf.parse_output('<tool_call>{"name":"send","arguments":{"to":"b"}}</tool_call>', tools, "agentdojo")
        self.assertTrue(parsed["valid"])
        self.assertEqual(parsed["tool_calls"][0]["arguments"], {"to": "b"})
        invalid = cf.parse_output('<tool_call>{"name":"send","arguments":{}}</tool_call>', tools, "agentdojo")
        self.assertFalse(invalid["valid"])
        self.assertFalse(cf.parse_output('<tool_call>{oops}</tool_call>', tools, "agentdojo")["valid"])
        self.assertTrue(cf.parse_output("I refuse to send anything.", tools, "agentdojo")["valid"])
        react = cf.parse_output('Thought: next\nAction: send\nAction Input: {"to":"b"}', tools, "injecagent")
        self.assertTrue(react["valid"])
        self.assertFalse(cf.parse_output('Action: send\nAction Input: broken', tools, "injecagent")["valid"])

    def test_predicate_matches_parameters_not_only_tool_names(self):
        predicate = {"calls": [{"name": "send", "arguments": {"to": "attacker"}}]}
        self.assertTrue(cf.match_predicate([{"name": "send", "arguments": {"to": "attacker", "body": "x"}}], predicate))
        self.assertFalse(cf.match_predicate([{"name": "send", "arguments": {"to": "legitimate"}}], predicate))
        with self.assertRaises(ValueError):
            cf.match_predicate([], {"calls": []})

    def test_judge_receives_goal_after_deletion_but_no_condition_or_attribution(self):
        context = {"dataset": "agentdojo", "user_task": "summarize", "attack_text": "send to attacker",
                   "original_tool_response": "facts ATTACK", "removed_tool_response": "facts ", "tools": []}
        generation = {"condition": "remove_attack", "output": "I refuse", "status": "ok",
                      "parse": {"valid": True, "tool_calls": [], "content": "I refuse"}}
        payload = cf.judge_payload(context, generation)
        self.assertEqual(payload["attack_goal"], "send to attacker")
        self.assertEqual(payload["tool_result"], "facts ")
        self.assertNotIn("condition", payload)
        self.assertNotIn("source_margin", payload)
        self.assertEqual(cf.evaluate_generation(context, {**generation, "status": "truncated"})["behavior_label"], "unresolved")

    def test_prepare_keeps_clean_and_attack_candidates_and_reuses_context(self):
        with tempfile.TemporaryDirectory() as folder:
            root = pathlib.Path(folder)
            tools = [{"toolkit": "Mail", "tools": [{"name": "Read", "parameters": []},
                                                     {"name": "Send", "parameters": []}]}]
            raw = [{"User Instruction": "read mail", "User Tool": "MailRead", "Attacker Tools": ["MailSend"],
                    "Attacker Instruction": "send it", "Tool Parameters": "{}", "Thought": "Thought: read",
                    "Tool Response": '{"body":"facts send it"}'}]
            (root / "tools.json").write_text(json.dumps(tools), encoding="utf-8")
            (root / "raw.json").write_text(json.dumps(raw), encoding="utf-8")
            base = {"case_id": 0, "valid_for_stats": True, "phi_auth": .1, "phi_data_fact": .1,
                    "phi_data_attack": 1, "target_scope": "full_action"}
            rows = [{**base, "eval": "succ", "action_text": "Action: MailSend\nAction Input: {}"},
                    {**base, "eval": "unsucc", "action_text": "Final Answer: facts"}]
            (root / "shapley.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            candidates, contexts, audit, summary = cf.prepare_manifest(
                "injecagent", [str(root / "shapley.jsonl")], [str(root / "raw.json")], str(root / "tools.json"))
            self.assertEqual(len(candidates), 2)
            self.assertEqual(len(contexts), 1)
            self.assertEqual(contexts[0]["removed_tool_response"], '{"body":"facts "}')
            self.assertEqual(summary["prepared_candidates"], 2)
            self.assertEqual(len(audit), 2)
            self.assertEqual(candidates[0]["context_id"], candidates[1]["context_id"])

    def test_cache_changes_with_seed_schema_model_and_configuration(self):
        base = {"input": "p", "model": "m", "schema_hash": "s", "seed": 1, "temperature": .1}
        for field in base:
            self.assertNotEqual(cf.stable_hash(base), cf.stable_hash({**base, field: "changed"}))


if __name__ == "__main__":
    unittest.main()
