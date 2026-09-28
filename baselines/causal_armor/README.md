# CausalArmor LOO attribution baseline for UCT-Trace

This repository's `baselines/causal_armor/causalarmor_loo_baseline.py` adapts **only the
attribution-based detection criterion** of CausalArmor to UCT-Trace's
fixed-action, offline classification setting. It does not implement
CausalArmor's privileged-action gate, sanitization, chain-of-thought masking,
or action regeneration. The source `D:\Github\causal-armor` is a third-party
implementation of the paper, not the authors' experimental release.

## Method

For each valid row of an existing UCT-Trace Shapley result file, the command
rebuilds the same original prompt and holds its target action fixed. It scores
three prompts with the **same target model**: complete prompt, original user
request content removed, and the selected polluted tool-result content
removed. The tool result is treated as one region containing both FACT and
ATTACK; the baseline never uses the ATTACK boundary. Role markers and other
messages remain in place. Each score is the mean teacher-forced target-token
log probability, so the LOO differences use the actual target token count:

```text
delta_user = logprob_full - logprob_without_user
delta_tool = logprob_full - logprob_without_tool_result
predict_success = delta_tool > delta_user - tau
```

The default `tau` is `0`. This is an **offline prediction of attack success**
from CausalArmor's attribution criterion, not a measurement of CausalArmor's
online attack success rate. Because this command performs no behavioral
verification, compare its classification metrics primarily with UCT-Trace's
Shapley-only prediction. The summary includes both methods on the identical
valid sample intersection.

## Requirements on the experiment computer

Use the Python and PyTorch environment required by this repository and make
the model weights named by `model_info.model_id` in the selected JSON config
accessible. The checked-in Qwen configuration contains a `/root/...` path
from another machine; set it to a path or model ID available on yours. Run
commands from the `D:\Github\Attention-Tracker` repository root. The LOO
command needs no judge API or CausalArmor proxy service.

## Commands

Use Shapley files generated from the **same source data and target model**.
For InjecAgent, pass the paired success and failure manifests together with
their matching generated-case files; this produces a meaningful two-class
classification summary. The input and output arguments should be adjusted
for each model.

### InjecAgent

```powershell
python baselines/causal_armor/causalarmor_loo_baseline.py `
  --dataset injecagent `
  --shapley-manifest result/injecagent_qwen3_dh_attack_20260809/results.action_shapley.jsonl result/injecagent_qwen3_dh_clean_20260809/results.action_shapley.jsonl `
  --input data/injecagent/qwen3-8b/test_cases_dh_base_attack.jsonl data/injecagent/qwen3-8b/test_cases_dh_base_clean.jsonl `
  --tools data/injecagent/injecagent_data/tools.json `
  --model-config configs/model_configs/qwen3_8b-attn_config.json `
  --output-dir result/causalarmor_loo_injecagent_qwen3
```

### AgentDojo

```powershell
python baselines/causal_armor/causalarmor_loo_baseline.py `
  --dataset agentdojo `
  --shapley-manifest result/agentdojo_fact_longer_than_attack_qwen_8b/results.shapley.jsonl `
  --input-root data/agentdojo/runs/qwen3-8b `
  --model-config configs/model_configs/qwen3_8b-attn_config.json `
  --output-dir result/causalarmor_loo_agentdojo_qwen3
```

The manifests determine the paired sample set. For InjecAgent, each manifest
must match the input in the same argument position; `sample_id` prefixes its
source-group number so success and failure cases with the same original
`case_id` stay distinct. Do not point the command at Shapley results from a
different model or differently selected data subset.
`--limit N` can restrict a later diagnostic run; omit it for the complete
comparison. `--tau` changes the detection threshold. The default long-context
switch and KV-cache chunk sizes match the AgentDojo Shapley script and can be
set with `--full-forward-token-threshold`, `--streaming-chunk-size`, and
`--prefill-chunk-size`.

## Output

The output directory receives only two new files:

- `results.causalarmor_loo.jsonl`: one row per selected Shapley sample, with
  identity, original fixed action, target type and token count, full and two
  ablated log probabilities, normalized LOO differences, margin, prediction,
  gold label, runtime, and any processing error. A failed row has
  `valid_for_stats=false` and is excluded from metrics.
- `results.causalarmor_loo.summary.json`: selection counts, runtime summary,
  and accuracy, precision, recall, FPR, and F1 for LOO and Shapley-only on
  the **same successfully scored rows**. Metrics are given for all fixed
  actions and separately for the tool-action subset.

No existing Shapley, attention, judge, or trajectory files are overwritten.
Run once for each dataset-model configuration. These commands and the new
baseline have not been executed against model weights in the current
environment; experimental results must be obtained on the later machine.
