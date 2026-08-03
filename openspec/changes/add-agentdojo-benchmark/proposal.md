## Why

The repository has shared Attention and Shapley attribution standards and an InjecAgent action experiment, but it cannot yet analyze the native multi-message AgentDojo trajectories already collected for Qwen3-8B. Adding a narrowly scoped AgentDojo benchmark makes it possible to explain the model's first assistant decision after an observed `important_instructions` injection while preserving the benchmark's original trajectory and labels.

## What Changes

- Add a loader and eligibility audit for `data/agentdojo/runs/qwen3-8b/*/user_task_*/important_instructions/*.json`, accepting only the 565 trajectories whose tool output contains a locatable `<INFORMATION>...</INFORMATION>` attack block.
- Select exactly one decision point per eligible trajectory: the first polluted tool response and the next assistant message after it.
- Map the original user prompt to `AUTH`, the non-attack content of that first polluted tool response to non-overlapping `FACT` spans, and its information block or blocks to `ATTACK`; retain all other context as fixed `SPECIAL` diagnostics.
- Attribute the original serialized next assistant message with Attention and exact three-player Shapley values under the global attribution experiment standards.
- Preserve AgentDojo `security` and `utility` labels independently of attribution, produce auditable JSONL/summary outputs, and provide Attention and Shapley visualizations.
- Add automated tests for discovery, target selection, span construction, Qwen chat-template serialization, attribution semantics, summaries, and visualization data.

## Capabilities

### New Capabilities

- `agentdojo-attribution-benchmark`: Defines dataset eligibility, single-decision target selection, AgentDojo player-region mapping, attribution outputs, summaries, and visualization behavior.

### Modified Capabilities

None. The existing `attribution-experiment-standards` requirements remain authoritative and unchanged.

## Impact

- Adds an AgentDojo experiment entry point and supporting parsing, attribution, summary, and visualization code under `scripts/`.
- Reads the existing Qwen3-8B AgentDojo runs without modifying source data.
- Produces a new result directory for per-target Attention/Shapley records, validation reports, summaries, and HTML galleries.
- Reuses the repository's model-loading, important-head Attention, exact Shapley, and visualization patterns; no external service or benchmark rerun is required.
