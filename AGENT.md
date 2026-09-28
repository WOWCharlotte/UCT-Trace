# Agent Guide: Environment Setup and First Run

Use this guide when helping someone set up and run UTC-Trace. The goal is to check the environment, install dependencies, validate the bundled data, and complete a first small run with reproducible commands. See [README.md](README.md) for the project overview and [README_zh.md](README_zh.md) for its Chinese version. Run every command from the repository root.

## Confirm the starting conditions

- Check the operating system, Python version, available memory, and GPU/CUDA status. The code uses Python 3.10+ syntax. The paper used PyTorch 2.5.1, Transformers 4.55.0, and CUDA 12.4; choose a PyTorch build compatible with the user's actual system.
- Identify the requested path: environment check, InjecAgent attribution, AgentDojo attribution, the Attention Tracker baseline, or behavioral judging. Start with the smallest useful run.
- Ask for the model weight directory or an accessible model ID when attribution is requested. The `/root/Qwen3-8B` value in `configs/model_configs/qwen3_8b-attn_config.json` is a path from another machine, not a portable default.
- Preserve existing virtual environments and model configurations. If the model location, GPU capability, or judge API details are unknown, complete checks that do not depend on them before requesting the missing information.

## 1. Set up and inspect Python

Use the user's preferred environment if one exists. Otherwise, create an isolated environment and install [requirements.txt](requirements.txt) and the test dependency. Example for Windows PowerShell:

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt pytest
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available())"
```

On Linux/macOS, activate it with `source .venv/bin/activate`. If PyTorch or CUDA installation fails, inspect the driver, Python version, and binary compatibility before changing packages. Attribution can be attempted on CPU, but an 8B model needs substantial memory; check GPU availability or an existing offload configuration first.

## 2. Run checks that do not load model weights

```bash
python -m pytest tests -q
python scripts/injecagent_action_experiment.py --validate_only --limit 1
```

The second command checks the bundled InjecAgent records against official cases and tool definitions. It writes `result/injecagent_qwen3_dh/results.action_shapley.summary.json`. Inspect `alignment.aligned` and `tool_coverage.covered`; an exit code alone does not establish that the data match. For AgentDojo, start with:

```bash
python scripts/agentdojo_attribution_experiment.py --audit-only
```

The default AgentDojo input is `data/agentdojo/runs/qwen3-8b/`; audit output goes to `result/agentdojo_qwen3_important_instructions/`. If corpus counts differ from expectations, inspect the input path and audit report first. Use `--allow-corpus-drift` only after confirming that a different corpus version is intended.

## 3. Configure a model and run one attribution sample

Model configurations live in `configs/model_configs/`. Set `model_info.model_id` to a local weight directory or a model ID the user can access; do not guess the download location. Prefer a separate local copy named `<model_name>_config.json` and select it with `--model_name <model_name>`. Check that the provider, model name, and device settings match the model in use.

```bash
python scripts/injecagent_action_experiment.py --model_name qwen3_8b-attn --limit 1
```

This example works only when the `model_id` in `qwen3_8b-attn_config.json` is accessible. Inspect `result/injecagent_qwen3_dh/results.action_shapley.jsonl`, `results.action_attention.jsonl`, and the summary JSON. Confirm that the expected records are valid; individual failures may be recorded in the output. A `--limit 1` run is not a paper-level result.

AgentDojo selects its model through `--model-config`:

```bash
python scripts/agentdojo_attribution_experiment.py --model-config configs/model_configs/qwen3_8b-attn_config.json --limit 1 --use-cache
```

Run this only after checking the config path and model weights. If the user wants only tool actions or long FACT samples, inspect the filtering options in the script's `--help`.

## 4. Optionally run the behavioral judge

Audit attribution records locally first; no API credentials are needed:

```bash
python scripts/llm_judge_attack_behavior.py --mode audit --dataset injecagent --input result/injecagent_qwen3_dh/results.action_shapley.jsonl --output result/judge.audit.jsonl --summary result/judge.audit.summary.json
```

`--mode judge` requires an OpenAI-compatible judge service. After the user supplies its details, configure `JUDGE_API_KEY`, `JUDGE_MODEL`, and, if needed, `JUDGE_BASE_URL`. Keep the key in an environment variable or the ignored `.env` file. Never print it or place it in a README, JSON config, log, or commit. Write judge results to separate paths so the attribution JSONL remains intact.

## Report the outcome

Tell the user which Python environment and commands were actually used, where the model configuration came from, which output files were produced, and what information is still needed. Distinguish a passed data check, a completed single-sample attribution, and a completed behavioral judgment. If a step fails, provide the first reproducible error, the command that produced it, and the next concrete fix.
