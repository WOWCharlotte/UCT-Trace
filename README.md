# UTC-Trace

[简体中文](README_zh.md)

UTC-Trace studies **unauthorized control transfer** in tool-using LLM agents. It explains whether an observed assistant response or tool call is supported by the user's instruction, task-relevant facts in a tool result, or injected instructions in that result. The manuscript calls the method **UCT-Trace**; this repository uses the requested name **UTC-Trace**.

The repository contains action-level experiments for **InjecAgent** and **AgentDojo**, an Attention Tracker baseline, a separate behavioral judge, and analysis scripts. Attribution requires access to model weights and token log probabilities; an API-only model is insufficient for that stage.

## Method at a glance

1. Split the context into `AUTH` (authorized user instruction), `FACT` (task-relevant tool-result content), and `ATTACK` (injected instructions). Hold the rest of the context fixed.
2. Keep the observed target action fixed and evaluate all eight coalitions of the three regions. Mask excluded regions at the token-embedding level and use the target's mean token log probability to compute exact Shapley contributions.
3. Flag an attribution candidate when `M_attack = φ_ATTACK − max(0, φ_AUTH) − max(0, φ_FACT) > 0`.
4. For candidates, check whether the observed action actually executes the attacker's requested task. An attack is predicted successful only when **ATTACK dominates attribution and behavioral execution is verified**. Mentioning or refusing the injected request is insufficient.

This is a post-hoc explanation workflow. It does not block agent actions. Attention Tracker remains available as a comparison baseline.

## Repository layout

```text
baselines/attention_tracker/   Original Attention Tracker model and detector code
configs/model_configs/        Model and attention-head configurations
data/injecagent/              InjecAgent inputs and generated action records
data/agentdojo/               AgentDojo trajectories
scripts/                      Attribution, behavior judging, analysis, and plots
tests/                        Unit and workflow tests
docs/                         Research notes and supporting material
result/                       Local outputs (Git-ignored)
```

## Quick start

Run these commands from the repository root. A Python environment with PyTorch and Transformers is required; GPU inference also requires a compatible CUDA setup.

```bash
python -m pip install -r requirements.txt
python -m pip install pytest
```

First, validate the bundled InjecAgent records without loading a model:

```bash
python scripts/injecagent_action_experiment.py --validate_only --limit 1
```

For an attribution smoke run, set the `model_id` in `configs/model_configs/qwen3_8b-attn_config.json` to a local or accessible Qwen3-8B model, then run:

```bash
python scripts/injecagent_action_experiment.py --model_name qwen3_8b-attn --limit 1
```

Results are written to `result/injecagent_qwen3_dh/`, including `results.action_shapley.jsonl` and `results.action_attention.jsonl`. To identify ATTACK-dominant candidates without an external judge:

```bash
python scripts/llm_judge_attack_behavior.py --mode audit --dataset injecagent --input result/injecagent_qwen3_dh/results.action_shapley.jsonl --output result/judge.audit.jsonl --summary result/judge.audit.summary.json
```

To verify execution, configure `JUDGE_API_KEY` and, if needed, `JUDGE_MODEL` and `JUDGE_BASE_URL` for an OpenAI-compatible judge, then use `--mode judge` with the same attribution input and new output paths. The judge writes separate files and does not overwrite attribution records.

For AgentDojo, start with `python scripts/agentdojo_attribution_experiment.py --audit-only`; see the script's `--help` for corpus and model options. Run the test suite with `python -m pytest tests -q`.

## License and upstream material

Original UTC-Trace contributions are licensed under the [MIT License](LICENSE). The included Attention Tracker baseline originates from [khhung-906/Attention-Tracker](https://github.com/khhung-906/Attention-Tracker), whose upstream README specifies **CC BY-NC 4.0**. That material and third-party datasets are **not relicensed by the MIT grant**. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before reuse.
