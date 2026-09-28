# Claude Guide: UTC-Trace

When helping a user set up this repository or run an experiment, follow the sequence in [AGENT.md](AGENT.md). Use [README.md](README.md) and [README_zh.md](README_zh.md) for project context. Run commands from the repository root.

## Default first-run sequence

1. Check Python, PyTorch, Transformers, and `torch.cuda.is_available()`. Reuse the user's environment when possible; otherwise create an isolated virtual environment and install `requirements.txt` and `pytest`.
2. Run `python -m pytest tests -q` and `python scripts/injecagent_action_experiment.py --validate_only --limit 1`. Inspect the validation JSON for case alignment and tool coverage.
3. Confirm that `model_info.model_id` in `configs/model_configs/qwen3_8b-attn_config.json` points to weights the user can access. `/root/Qwen3-8B` is not a portable path. When appropriate, create a separate local config instead of overwriting an existing one.
4. Start attribution with `python scripts/injecagent_action_experiment.py --model_name qwen3_8b-attn --limit 1`. If using a local config copy, change `--model_name` accordingly. Inspect the JSONL and summary files in `result/injecagent_qwen3_dh/`.
5. Audit attribution candidates locally with `--mode audit`. Run `--mode judge` only after the user provides a judge service and credentials. Never expose or commit API keys.

For AgentDojo, first run `python scripts/agentdojo_attribution_experiment.py --audit-only`, verify the corpus and model configuration, then try `--limit 1 --use-cache`. If the user wants only the Attention Tracker baseline, its entry points are under `baselines/attention_tracker/`; do not present a baseline detection result as a UTC-Trace behavioral judgment.

At the end, report which stages actually completed, their output paths, and the specific information needed for the next stage. Missing model weights should not prevent checks that do not load a model.
