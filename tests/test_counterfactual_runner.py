import copy
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import run_counterfactual_experiment as runner


class NextTokenModel(torch.nn.Module):
    device = torch.device("cpu")

    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 8)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.eye(8))
        self.calls = []

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, attention_mask, use_cache, past_key_values=None, **kwargs):
        self.calls.append((inputs_embeds.detach().clone(), past_key_values))
        ids = inputs_embeds.argmax(-1)
        logits = torch.full((*ids.shape, 8), -20.)
        logits.scatter_(2, ((ids + 1) % 8).unsqueeze(-1), 20.)
        return type("Output", (), {"logits": logits, "past_key_values": object() if use_cache else None})()


class CounterfactualRunnerTests(unittest.TestCase):
    def test_teacher_forcing_full_and_streaming_alignment(self):
        wrapped = type("Wrapped", (), {"model": NextTokenModel()})()
        full = runner.score_ids(wrapped, [1, 2], [3, 4, 5], threshold=100)
        streamed = runner.score_ids(wrapped, [1, 2], [3, 4, 5], threshold=1, chunk_size=2, prefill_size=1)
        self.assertAlmostEqual(full["mean_logprob"], 0., places=5)
        self.assertAlmostEqual(full["mean_logprob"], streamed["mean_logprob"], places=5)
        wrong = runner.score_ids(wrapped, [1, 2], [4, 5, 6], threshold=100)
        self.assertLess(wrong["mean_logprob"], -10)
        # None starts a new prefill; cached state is never reused between interventions.
        self.assertIsNone(wrapped.model.calls[-1][1])

    def test_attempt_log_retries_failures_and_keeps_history(self):
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "rows.jsonl"
            log = runner.AttemptLog(path, resume=False)
            log.append({"cache_key": "a", "status": "failed"})
            log.append({"cache_key": "b", "status": "ok"})
            restarted = runner.AttemptLog(path, resume=True)
            self.assertIsNone(restarted.cached("a"))
            self.assertEqual(restarted.cached("b")["status"], "ok")
            restarted.append({"cache_key": "a", "status": "ok"})
            self.assertEqual(len(runner.cf.read_rows(path)), 3)
            self.assertEqual(runner.cf.read_rows(path)[-1]["attempt"], 2)

    def test_generation_explicit_parameters_and_fresh_cache(self):
        class Tokenizer:
            eos_token_id, pad_token_id, model_max_length = 7, 0, 100
            def decode(self, ids, skip_special_tokens=True):
                return "Final Answer: safe"
        class Model:
            device = torch.device("cpu")
            config = type("Config", (), {"max_position_embeddings": 100})()
            generation_config = type("Config", (), {"eos_token_id": [7]})()
            def generate(self, **kwargs):
                self.kwargs = kwargs
                return torch.cat([kwargs["input_ids"], torch.tensor([[3, 7]])], dim=1)
        wrapped = type("Wrapped", (), {"model": Model(), "tokenizer": Tokenizer()})()
        result = runner.generate_ids(wrapped, [1, 2], 1, {"temperature": .1, "top_k": 50, "top_p": 1., "max_new_tokens": 5})
        self.assertEqual(result["status"], "ok")
        self.assertFalse(wrapped.model.kwargs["output_attentions"])
        self.assertEqual(wrapped.model.kwargs["temperature"], .1)
        self.assertNotIn("past_key_values", wrapped.model.kwargs)
        self.assertEqual(result["stop_reason"], "eos")
        capped = runner.generate_ids(wrapped, [1, 2], 1, {"temperature": .1, "top_k": 50, "top_p": 1., "max_new_tokens": 2})
        self.assertEqual(capped["status"], "truncated")
        self.assertEqual(capped["stop_reason"], "max_new_tokens")
        with self.assertRaises(ValueError):
            runner.generate_ids(wrapped, [1] * 99, 1, {"temperature": .1, "top_k": 50, "top_p": 1., "max_new_tokens": 5})

    def test_schema_export_refuses_wrong_version(self):
        with mock.patch("importlib.metadata.version", return_value="0.0.1"):
            with self.assertRaises(ValueError):
                runner.export_agentdojo_schemas()

    def test_generation_messages_only_change_audited_result(self):
        context = {"dataset": "agentdojo", "messages": [{"role": "system", "content": "system"},
                   {"role": "user", "content": "task"}, {"role": "tool", "content": "fact attack"}],
                   "polluted_tool_index": 2, "original_tool_response": "fact attack", "removed_tool_response": "fact "}
        original = copy.deepcopy(context)
        schemas = [{"name": "read", "description": "read", "parameters": {"type": "object"}}]
        messages = runner.generation_messages(context, "remove_attack", schemas)
        self.assertEqual(messages[2]["content"], "fact ")
        self.assertIn("read", messages[0]["content"])
        self.assertEqual(messages[1], context["messages"][1])
        self.assertEqual(context, original)

    def test_offline_end_to_end_pipeline_and_resume(self):
        """Exercise the real stages with a tiny local model, not experiment data."""
        class Tokenizer:
            eos_token_id, pad_token_id, model_max_length, chat_template = 0, 0, 8192, None
            def encode(self, text, add_special_tokens=False):
                return [ord(character) for character in text]
            def decode(self, ids, skip_special_tokens=True):
                return "".join(chr(value) for value in ids if value)
        class TinyModel(torch.nn.Module):
            device = torch.device("cpu")
            config = type("Config", (), {"max_position_embeddings": 8192})()
            generation_config = type("Config", (), {"eos_token_id": 0})()
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(256, 8)
                self.projection = torch.nn.Linear(8, 256)
            def get_input_embeddings(self):
                return self.embedding
            def forward(self, inputs_embeds, attention_mask, use_cache, past_key_values=None, **kwargs):
                return type("Output", (), {"logits": self.projection(inputs_embeds), "past_key_values": object() if use_cache else None})()
            def generate(self, input_ids, **kwargs):
                tail = torch.tensor([[ord(char) for char in "Final Answer: facts"] + [0]])
                return torch.cat([input_ids, tail], dim=1)
        wrapped = type("Wrapped", (), {"model": TinyModel(), "tokenizer": Tokenizer()})()
        provenance = {"model_config": {"model_info": {"name": "tiny-test-model"}}}
        # The Windows app sandbox does not permit atomic rename in its redirected
        # system temp directory. Keep this file-I/O integration fixture in the repo cache.
        cache = pathlib.Path(__file__).resolve().parents[1] / ".pytest_cache"
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as folder:
            root = pathlib.Path(folder)
            tools = [{"toolkit": "Mail", "tools": [{"name": "Read", "parameters": []}, {"name": "Send", "parameters": []}]}]
            raw = [{"User Instruction": "read", "User Tool": "MailRead", "Attacker Tools": ["MailSend"],
                    "Attacker Instruction": "send", "Tool Parameters": "{}", "Thought": "Thought: read",
                    "Tool Response": '{"body":"facts send"}'}]
            rows = [{"case_id": 0, "valid_for_stats": True, "phi_auth": .1, "phi_data_fact": .1,
                     "phi_data_attack": 1, "action_text": "Action: MailSend\nAction Input: {}", "target_scope": "full_action"}]
            for name, data in (("tools.json", tools), ("raw.json", raw)):
                (root / name).write_text(json.dumps(data), encoding="utf-8")
            (root / "shapley.jsonl").write_text(json.dumps(rows[0]), encoding="utf-8")
            common = ["cf", "--dataset", "injecagent", "--output-dir", str(root / "out"),
                      "--seeds", "1", "2", "--resume", "--tools", str(root / "tools.json")]
            for stage in ("prepare", "validate", "score", "generate", "evaluate", "analyze"):
                argv = common + ["--stage", stage, "--model-config", "fake.json", "--judge-mode", "deterministic",
                                 "--raw-input", str(root / "raw.json"), "--shapley-input", str(root / "shapley.jsonl")]
                with mock.patch.object(sys, "argv", argv), mock.patch.object(runner, "load_model", return_value=(wrapped, provenance)):
                    runner.main()
            summary = json.loads((root / "out/summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["support"]["actions"], 1)
            self.assertEqual(summary["behavior"]["paired_contexts"], 1)
            self.assertEqual(summary["behavior"]["paired_effect"]["mean"], 0.)
            self.assertEqual(summary["coverage"]["expected_generations"], 4)
            initial = runner.cf.read_rows(root / "out/generations.jsonl")
            with mock.patch.object(sys, "argv", common + ["--stage", "generate", "--model-config", "fake.json"]), mock.patch.object(runner, "load_model", return_value=(wrapped, provenance)):
                runner.main()
            self.assertEqual(runner.cf.read_rows(root / "out/generations.jsonl"), initial)


if __name__ == "__main__":
    unittest.main()
