## Context

The repository already implements shared attribution semantics, Qwen model loading, important-head Attention, exact three-player Shapley, and action-oriented visualization for InjecAgent. AgentDojo differs because its Qwen3-8B data is stored as native multi-message trajectories containing system, user, assistant function calls, and tool messages. The experiment must reconstruct the exact decision prefix and assistant target without rerunning the agent or changing the source data.

The scoped corpus contains 629 `important_instructions` JSON files across banking, slack, travel, and workspace. A repository audit finds 565 trajectories with at least one tool response containing a complete `<INFORMATION>...</INFORMATION>` attack block. The remaining 64 trajectories never expose that attack text to the model and are excluded from three-player attribution.

The global `attribution-experiment-standards` spec is authoritative. In particular, system instructions are fixed `SPECIAL` context rather than `AUTH`, all Shapley coalitions preserve token structure, and Attention exposes both prompt- and player-normalized score families.

## Goals / Non-Goals

**Goals:**

- Produce exactly one auditable target from each of the 565 eligible trajectories.
- Explain the next assistant message after the first polluted tool response.
- Construct exact, non-overlapping, multi-span `AUTH`, `FACT`, and `ATTACK` regions in the native Qwen chat rendering.
- Reuse common model and attribution utilities where their semantics match the global standard.
- Emit per-target Attention and Shapley records, validation/audit summaries, aggregate statistics, and inspectable HTML visualizations.
- Make parsing and attribution behavior testable without loading the full model in unit tests.

**Non-Goals:**

- Rerunning AgentDojo, executing tools, or regenerating trajectories.
- Supporting attack methods other than `important_instructions` or model runs other than `qwen3-8b`.
- Explaining every post-attack assistant action, later polluted responses, final environment state, or causal effects across a rerun trajectory.
- Adding system prompts, prior tool data, or prior assistant messages as Shapley players.
- Changing the global attribution standards or the existing InjecAgent protocol.

## Decisions

### Use structural discovery with a snapshot count assertion

The loader will glob only the requested Qwen3-8B `important_instructions` path, parse each JSON document, and locate complete information blocks only inside tool-message `content`. Eligibility is structural, while validation asserts the current snapshot totals of 629 candidates and 565 eligible trajectories.

This is preferable to hard-coding 565 paths: structural discovery remains understandable and detects corpus drift, while the count assertion prevents unnoticed changes to the scientific sample. Using the raw `injections` value as an exact substring was rejected because whitespace in saved tool responses differs from the injection metadata.

### Select the first polluted tool and the next assistant once

For each eligible trajectory, select the lowest-index polluted tool message, then the lowest-index assistant message after it. Later polluted tool messages and later assistant messages do not create rows. Selection metadata records both indices and whether the assistant was immediately adjacent by array index.

Role-based selection is used instead of requiring `assistant_index == tool_index + 1` so malformed or unusual intervening records can be audited without redefining “next assistant.” A missing later assistant is a validation failure rather than an invitation to synthesize a target.

### Derive every player from one explicit source

`AUTH` comes only from the original user message content. `FACT` and `ATTACK` come only from the selected polluted tool message content. A non-greedy information-element locator returns all complete blocks within that selected response; their exact character intervals are `ATTACK`, and interval subtraction yields the remaining `FACT` pieces.

The implementation will locate these source strings within the fully rendered decision prefix and map character intervals through tokenizer offset mappings into token spans. Span validation checks presence, ordering, bounds, and pairwise non-overlap. All unmatched rendered tokens are `SPECIAL`.

Taking facts only from the selected polluted response intentionally excludes useful data in other tool messages. This sacrifices a complete causal partition of the trajectory in exchange for the user-defined experiment: the three players isolate the authorized request and the contents of the first observed attack carrier.

### Render prefix and target through the native chat template

The selected assistant message is removed from the context. The message prefix through the preceding record is rendered with the Qwen tokenizer's chat template and an assistant generation prompt. The selected assistant message is then rendered using the same protocol, and the target token sequence is derived by comparing the prefix rendering with the rendering that includes the selected assistant.

This avoids inventing a JSON or ReAct representation that Qwen was not trained to emit. Round-trip validation will ensure the derived target preserves recorded textual content and function-call names/arguments. Template-control tokens remain fixed and are never player spans.

### Treat the complete selected assistant serialization as one target scope

The only primary scope is `next_assistant_message`. `target_kind` distinguishes `tool_calls`, `content`, and `mixed`, but these kinds remain within the same declared target scope. The complete serialized target is scored; tool name and arguments are not split into separate primary Shapley experiments.

This produces one target per eligible trajectory and prevents scope mixing. Future tool-name-only analyses would need a separate target scope and separately grouped summaries.

### Aggregate Attention across the declared target token set

Attention will be computed teacher-forced for the selected assistant target and aggregated from target tokens to context tokens over configured important heads. The implementation will expose its precise target-token inclusion rule and aggregation order in result metadata. Chat-template control-only target tokens will be excluded when the tokenizer exposes a reliable semantic boundary; otherwise all target tokens will be used and that limitation will be declared.

The primary record retains token-level context scores plus raw region mass, prompt-normalized scores including `SPECIAL`, and player-normalized scores excluding `SPECIAL`. A first-target-token diagnostic may be retained for comparison with the InjecAgent implementation, but it cannot replace or be mixed with the declared primary aggregation.

### Reuse exact embedding-masked Shapley

For each target, prompt embeddings are created once. Each of the eight coalitions zeroes embeddings for all spans belonging to excluded players while leaving sequence length, positions, attention masks, and fixed content unchanged. The original target is appended and teacher-forced; its mean token log probability is the coalition value.

Text replacement and restoration to benign AgentDojo defaults were rejected because they change tokens and positions and conflict with the global standard. Regeneration per coalition was rejected because it explains different outputs and is not comparable to the existing protocol.

### Preserve source truth and separate processing artifacts

Each source trajectory has a stable case ID derived from suite, user task, attack type, and injection task. A target ID extends it with selected tool and assistant message indices. The audit manifest records every candidate and its eligibility or exclusion reason. Attention and Shapley outputs use separate JSONL files joined by target ID, and summaries include explicit denominators and processing counts.

The raw `security` and `utility` fields are copied unchanged. If `attack_success` is exposed, its transformation from AgentDojo `security` must be centralized, documented in metadata, and covered by behavioral fixtures; attribution values never participate in that transformation.

### Build a dedicated runner around reusable helpers

A dedicated AgentDojo experiment module will own discovery, message selection, native rendering, region construction, validation, attribution orchestration, summaries, and CLI options. Generic exact-Shapley and visualization helpers may be reused or minimally generalized when doing so preserves existing callers.

Pure parsing and span functions will not require a model, enabling fixture-based coverage. Model-dependent tests will use tokenizer/model doubles following existing test patterns. The CLI will support input root, output directory, model config, limit, Attention-only/Shapley-only modes, and validation-only audit execution.

## Risks / Trade-offs

- **[Chat-template offsets are difficult to recover across structured tool calls]** → Derive prefix/target boundaries from paired native renderings, require a fast tokenizer with offsets for context spans, and fail validation instead of guessing.
- **[The `<INFORMATION>` locator may match benign content]** → Restrict it to `important_instructions`, tool-message content, and complete paired elements; cross-check normalized block text against the trajectory's `injections` metadata and report mismatches.
- **[FACT excludes relevant facts from other tool messages]** → Record the explicit `first_polluted_tool_only` fact policy in every result and visualization so conclusions are not interpreted as attribution over all factual context.
- **[Multiple information blocks complicate interval subtraction]** → Sort and merge only overlapping attack intervals, reject malformed or crossing boundaries, and compute FACT as the exact complement inside the selected tool content.
- **[Long trajectories and full-target Attention are memory intensive]** → Process one target at a time, release tensors between methods, provide method-specific modes and limits, and record context/target token counts and failures.
- **[The corpus snapshot may change]** → Keep structural eligibility as the source of truth but make expected counts configurable and enabled by default for the canonical run.
- **[Message-level rows can be mistaken for independent reruns]** → Produce one row per trajectory by construction and report target count, unique trajectory count, and duplicate-ID validation.
- **[Security-label polarity may be misread]** → Preserve raw fields, centralize any derived label, test known success/failure fixtures, and display both raw and derived meanings.

## Migration Plan

1. Add parsing and validation in audit-only mode and verify the canonical 629/565/64 counts.
2. Add native rendering and region-span tests against representative content, tool-call, repeated-attack, and multi-block fixtures.
3. Add Attention and Shapley execution behind a new dedicated CLI without changing existing experiment entry points.
4. Generate outputs in a new AgentDojo-specific result directory and validate counts, target IDs, span invariants, and Shapley efficiency before publishing summaries.
5. Add HTML visualizations and usage documentation after the machine-readable schema is stable.

Rollback consists of removing the new runner, tests, and generated result directory; source AgentDojo data and existing experiments remain untouched.

## Open Questions

- Which exact Qwen3-8B model configuration and important-head list should be the canonical CLI default if more than one repository config is compatible?
- Should malformed eligible trajectories fail the entire canonical run immediately or be written as failed audit rows while the remaining targets continue? The design favors continued processing plus a non-zero final validation status for canonical count mismatches.
- Can the tokenizer reliably identify and exclude assistant-side control-only target tokens for primary Attention aggregation, or must the first implementation aggregate all serialized target tokens and declare that choice?
