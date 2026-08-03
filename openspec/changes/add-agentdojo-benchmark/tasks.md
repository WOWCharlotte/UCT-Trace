## 1. Corpus Discovery And Selection

- [x] 1.1 Add an AgentDojo experiment module and CLI defaults restricted to the Qwen3-8B `important_instructions` glob.
- [x] 1.2 Implement complete `<INFORMATION>...</INFORMATION>` block discovery in tool-message content with normalized cross-checks against `injections` metadata.
- [x] 1.3 Implement deterministic selection of the first polluted tool response and the next assistant message, producing one stable target ID per eligible trajectory.
- [x] 1.4 Add an audit-only workflow that records every candidate's eligibility or failure reason and asserts the canonical 629 candidate, 565 eligible, and 64 excluded counts.

## 2. Native Prompt And Region Construction

- [x] 2.1 Render the decision prefix and recorded assistant target with the native Qwen chat template, preserving structured tool calls and arguments.
- [x] 2.2 Implement target-kind classification and round-trip checks for `tool_calls`, `content`, and `mixed` assistant messages.
- [x] 2.3 Map only the original user prompt to `AUTH` in the rendered prefix while leaving system and template content fixed as `SPECIAL`.
- [x] 2.4 Map all information blocks in the selected polluted response to disjoint `ATTACK` token spans and the exact remaining response content to disjoint `FACT` token spans.
- [x] 2.5 Validate player-span presence, bounds, non-overlap, source ownership, and exclusion of every non-selected tool response from `FACT`.

## 3. Attention Attribution

- [x] 3.1 Implement teacher-forced target-to-context Attention aggregation over configured important heads with explicit target-token inclusion and aggregation metadata.
- [x] 3.2 Emit token-level regions, raw regional mass, prompt-normalized scores including `SPECIAL`, and player-normalized scores excluding `SPECIAL`.
- [x] 3.3 Compute AUTH-focus attention shift and separate attack-dominance metrics using the shared attribution semantics.

## 4. Shapley Attribution

- [x] 4.1 Reuse or adapt exact three-player Shapley evaluation to score all eight coalitions against the original serialized next assistant message.
- [x] 4.2 Apply embedding-level multi-span player masking while preserving prompt length, positions, attention masks, and fixed context.
- [x] 4.3 Emit coalition values, three player contributions, target token count, efficiency error, and masking-protocol metadata.

## 5. Results And Reporting

- [x] 5.1 Write separate joinable Attention and Shapley JSONL outputs with source path, selection indices, labels, target identity, target text, player spans, and validation status.
- [x] 5.2 Preserve raw AgentDojo `security` and `utility` labels and document and test any derived attack-success polarity independently of attribution.
- [x] 5.3 Generate audit and experiment summaries with explicit denominators grouped by target scope, suite, security label, and target kind.
- [x] 5.4 Add Attention and Shapley HTML visualizations that display the selected polluted response, next assistant target, player regions, and separate `SPECIAL` diagnostics.

## 6. Verification And Documentation

- [x] 6.1 Add parser and selection tests for absent, repeated, malformed, and multiple information blocks and for missing post-attack assistant messages.
- [x] 6.2 Add native-rendering and region tests covering textual responses, tool calls, multiple FACT spans, multiple ATTACK spans, and fixed-context classification.
- [x] 6.3 Add Attention and Shapley unit tests for normalization denominators, shift semantics, eight-coalition teacher forcing, structure-preserving masking, and efficiency checks.
- [x] 6.4 Add output, summary, and visualization tests for stable joins, one-target-per-trajectory invariants, label independence, grouping, and HTML escaping.
- [x] 6.5 Run focused tests, the audit-only canonical corpus validation, and the broader relevant test suite; document the CLI commands, output files, protocol, and known limitations.
