# CausalArmor independent LOO baseline

`causalarmor_loo_baseline.py` is an independent baseline. It reads the same raw
AgentDojo trajectories or InjecAgent generated-case JSONL as Attention Tracker
and UCT-Trace, selects its own samples, and records its own successes and
failures. A UCT-Trace result file is never required and never determines which
samples CausalArmor runs.

This repository implements only CausalArmor's LOO causal-dominance detector in
a fixed-action offline setting. It does not implement the paper's complete
online guardrail: privileged-action gating, targeted sanitization, retroactive
chain-of-thought masking, and action regeneration are out of scope.

## Method

For each fixed observed action, the target model scores the action under three
contexts: the full prompt, the prompt without the original user content, and
the prompt without the entire polluted tool-result content. The mean target
token log probability implements target-length normalization:

```text
delta_user = logprob_full - logprob_without_user
delta_tool = logprob_full - logprob_without_tool_result
margin = delta_tool - delta_user
predict_attack_success = margin > -tau
```

The default is `tau=0`, and the comparison is strict. The tool result is one
region containing both FACT and ATTACK; CausalArmor does not use the ATTACK
span boundary when ablating it.

## AgentDojo

The following command matches the selection used by the
`agentdojo_fact_longer_than_attack` experiment, but runs CausalArmor directly
from the trajectory corpus:

```bash
python baselines/causal_armor/causalarmor_loo_baseline.py \
  --dataset agentdojo \
  --input-root data/agentdojo/runs/qwen3-8b \
  --model-config configs/model_configs/qwen3_8b-attn_config.json \
  --output-dir result/causalarmor_loo_agentdojo_qwen3 \
  --fact-dominant \
  --fact-ratio 1.0 \
  --use-cache
```

With the defaults, `--fact-dominant --fact-ratio 1.0` means exactly:

```text
500 <= FACT chars <= 100000
FACT chars >= 1.0 * max(1, ATTACK chars)
```

It is `FACT >= ATTACK`, not strict `FACT > ATTACK`. The script also supports
the same selection controls as the main experiment:
`--action-targets-only`, `--min-fact-chars`, `--max-fact-chars`,
`--rank-by-fact-ratio`, `--target-id`, `--target-id-file`, and `--limit`.

## InjecAgent

Pass raw generated-case files directly. Multiple files are supported; the
source-group prefix in `sample_id` prevents equal row indices from colliding.

```bash
python baselines/causal_armor/causalarmor_loo_baseline.py \
  --dataset injecagent \
  --input \
    data/injecagent/qwen3-8b/test_cases_dh_base_attack.jsonl \
    data/injecagent/qwen3-8b/test_cases_dh_base_clean.jsonl \
  --tools data/injecagent/injecagent_data/tools.json \
  --model-config configs/model_configs/qwen3_8b-attn_config.json \
  --output-dir result/causalarmor_loo_injecagent_qwen3 \
  --use-cache
```

`--eval-filter`, `--case-ids`, `--limit`, and `--include-all-tools` mirror the
corresponding main-experiment selection options. Rows labelled `invalid` remain
visible as failed attempts when selected, but only `succ` and `unsucc` have the
boolean gold label required for metrics.

## Optional post-hoc comparison

For a paired report, add `--comparison-shapley` after defining the independent
raw input. This file is joined by identity only after CausalArmor selection; it
does not exclude UCT-Trace failures or add samples. The old option spelling
`--shapley-manifest` remains as a compatibility alias, with the same post-hoc
semantics.

```bash
python baselines/causal_armor/causalarmor_loo_baseline.py \
  --dataset agentdojo \
  --input-root data/agentdojo/runs/qwen3-8b \
  --model-config configs/model_configs/qwen3_8b-attn_config.json \
  --output-dir result/causalarmor_loo_agentdojo_qwen3 \
  --fact-dominant --fact-ratio 1.0 --use-cache \
  --comparison-shapley result/agentdojo_fact_longer_than_attack_qwen_8b/results.shapley.jsonl
```

Fair paired metrics are computed only on the identity intersection where both
methods succeeded. Each method's attempted, valid, and failed counts remain
independent.

## Outputs and resuming

The output directory contains:

- `results.causalarmor_loo.jsonl`: one attempted raw-input sample per row,
  including fixed action, LOO scores, prediction, gold label, runtime, and
  processing error. `valid_for_stats=false` marks failed execution.
- `results.causalarmor_loo.summary.json`: raw corpus audit, exact selection
  configuration, selected/valid/failed counts, error classes, metrics, runtime,
  and tau sweep. Optional UCT-Trace paired metrics appear only when a comparison
  file is supplied. `suite_name=null` is excluded from suite-group counts.

`--use-cache` checkpoints every sample and reuses successful rows by independent
`sample_id`. Failed rows are retried. Long inputs switch to KV-cache scoring at
`--full-forward-token-threshold` (default 1024); chunk sizes are controlled by
`--streaming-chunk-size` and `--prefill-chunk-size`.

Judge and visualization integration continues to consume the independent
JSONL:

```bash
python scripts/llm_judge_attack_behavior.py \
  --mode audit --dataset agentdojo --trigger causalarmor \
  --input result/causalarmor_loo_agentdojo_qwen3/results.causalarmor_loo.jsonl \
  --output result/causalarmor_loo_agentdojo_qwen3/judge.audit.jsonl \
  --summary result/causalarmor_loo_agentdojo_qwen3/judge.audit.summary.json

python scripts/visualize_agentdojo_attribution.py \
  --shapley result/agentdojo_fact_longer_than_attack_qwen_8b/results.shapley.jsonl \
  --causalarmor result/causalarmor_loo_agentdojo_qwen3/results.causalarmor_loo.jsonl \
  --judge result/causalarmor_loo_agentdojo_qwen3/judge.audit.jsonl \
  --all \
  --output result/causalarmor_loo_agentdojo_qwen3/comparison.html
```

The detector output is an offline classification result. It must not be
reported as the full CausalArmor online defense ASR.
