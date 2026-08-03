## ADDED Requirements

### Requirement: AgentDojo Input Scope Is Restricted And Audited
The benchmark SHALL discover JSON inputs only from `data/agentdojo/runs/qwen3-8b/*/user_task_*/important_instructions/*.json`. It SHALL accept a trajectory only when at least one tool message contains a locatable complete `<INFORMATION>...</INFORMATION>` block, and the current dataset audit MUST yield exactly 565 eligible trajectories from 629 candidates.

#### Scenario: Discover the current benchmark corpus
- **WHEN** the benchmark audits the repository's Qwen3-8B AgentDojo data
- **THEN** it SHALL report 629 candidate trajectories, 565 eligible trajectories, and 64 exclusions whose reason is `no_information_attack_block`

#### Scenario: Ignore other attack methods and model runs
- **WHEN** AgentDojo data exists outside the Qwen3-8B `important_instructions` directories
- **THEN** the benchmark SHALL NOT include that data in candidate or eligible counts

### Requirement: Each Trajectory Selects One Polluted Tool Response
For each eligible trajectory, the benchmark SHALL select the earliest tool message whose content contains a complete `<INFORMATION>...</INFORMATION>` block. Later polluted tool messages MUST NOT create additional experiment samples or contribute player content.

#### Scenario: Trajectory contains repeated polluted responses
- **WHEN** more than one tool message in a trajectory contains an information attack block
- **THEN** the benchmark SHALL select only the earliest polluted tool message
- **THEN** the trajectory SHALL still produce exactly one experiment sample

### Requirement: Each Trajectory Selects The Next Assistant Message
The explained object SHALL be the first assistant message occurring after the selected polluted tool response. The benchmark MUST NOT explain any later assistant message from that trajectory.

#### Scenario: Select the next assistant by role
- **WHEN** an eligible trajectory contains one or more assistant messages after the selected polluted tool response
- **THEN** the benchmark SHALL select the assistant message with the lowest later message index
- **THEN** it SHALL record `target_scope=next_assistant_message`

#### Scenario: No assistant follows the polluted response
- **WHEN** an otherwise eligible trajectory has no assistant message after the selected polluted tool response
- **THEN** validation SHALL fail for that trajectory with `no_post_attack_assistant`
- **THEN** the benchmark SHALL NOT silently select an earlier or synthetic target

### Requirement: AgentDojo Players Use Exact Source Boundaries
The benchmark SHALL use exactly `AUTH`, `FACT`, and `ATTACK` as attribution players. `AUTH` SHALL contain only the original user prompt. `FACT` SHALL contain only the selected polluted tool message content excluding every located information attack block. `ATTACK` SHALL contain every complete information attack block in that selected tool message.

#### Scenario: Split factual content around one attack
- **WHEN** the selected tool content is `FACT_PREFIX + ATTACK + FACT_SUFFIX`
- **THEN** `FACT` SHALL contain separate spans for `FACT_PREFIX` and `FACT_SUFFIX`
- **THEN** `ATTACK` SHALL contain the attack span
- **THEN** no `FACT` span SHALL overlap the `ATTACK` span

#### Scenario: Exclude other tool responses from FACT
- **WHEN** the trajectory contains tool messages before or after the selected polluted tool response
- **THEN** their content SHALL NOT be assigned to `FACT`

#### Scenario: Selected response contains multiple attack blocks
- **WHEN** the selected polluted tool response contains multiple complete information attack blocks
- **THEN** all blocks SHALL be represented as disjoint `ATTACK` spans
- **THEN** the remaining response segments SHALL be represented as disjoint `FACT` spans

### Requirement: Fixed Context Is Preserved As SPECIAL
System messages, tool definitions and schemas, chat-template syntax, role and control tokens, message delimiters, prior assistant messages, prior non-selected tool responses, and tool metadata SHALL remain fixed in the model context and SHALL be classified as `SPECIAL` diagnostics rather than attribution players.

#### Scenario: Build the decision context
- **WHEN** the benchmark renders the trajectory prefix ending immediately before the selected assistant target
- **THEN** it SHALL retain the original message order and fixed context
- **THEN** only original user-prompt tokens, selected-response fact tokens, and selected-response attack tokens SHALL belong to player regions

### Requirement: Target Serialization Uses The Native Qwen Chat Protocol
The benchmark SHALL serialize the original selected assistant message with the Qwen3-8B tokenizer's native chat template. The target SHALL preserve the recorded assistant content, complete tool names, and complete tool arguments as applicable.

#### Scenario: Explain a tool-call assistant message
- **WHEN** the selected assistant message contains tool calls
- **THEN** `target_kind` SHALL be `tool_calls` or `mixed` as appropriate
- **THEN** `target_text` SHALL include the complete native serialization of every selected-message tool call and its arguments

#### Scenario: Explain a textual assistant message
- **WHEN** the selected assistant message contains content and no tool calls
- **THEN** `target_kind` SHALL be `content`
- **THEN** the complete serialized content SHALL be used as the explained target

### Requirement: Attention Follows Shared Attribution Semantics
The benchmark SHALL compute Attention from the selected assistant target to the rendered decision context using the configured important heads. It SHALL report raw regional mass, prompt-normalized scores including `SPECIAL`, and player-normalized scores over `AUTH + FACT + ATTACK`, and SHALL compute `auth_focus_score`, `attention_shift`, and `attention_attack_dominant` according to the global attribution experiment standards.

#### Scenario: Write Attention output
- **WHEN** Attention is successfully computed for an eligible target
- **THEN** the result SHALL retain token ranges and scores for `AUTH`, `FACT`, `ATTACK`, and `SPECIAL`
- **THEN** it SHALL declare the target-token and head aggregation method in source metadata
- **THEN** `SPECIAL` SHALL NOT appear in the player-normalized denominator

### Requirement: Shapley Explains The Original Next Assistant Message
The benchmark SHALL evaluate all eight coalitions over `AUTH`, `FACT`, and `ATTACK` by teacher-forcing the original serialized next assistant message and using mean target-token log probability as the coalition value. Excluding a player MUST use embedding-level masking that preserves prompt length, token positions, attention masks, fixed context, and tool schemas.

#### Scenario: Compute an exact three-player explanation
- **WHEN** Shapley attribution runs for an eligible target
- **THEN** it SHALL score the same original target for all eight coalitions
- **THEN** it SHALL emit `phi_auth`, `phi_data_fact`, `phi_data_attack`, coalition values, target token count, and efficiency error

### Requirement: Source Labels Remain Independent Of Attribution
The benchmark SHALL preserve the source trajectory's `security` and `utility` values and SHALL derive any convenience `attack_success` field only from the documented AgentDojo security-label semantics. Attention or Shapley values MUST NOT change source labels or determine attack success.

#### Scenario: Emit labels with attribution
- **WHEN** a result row is written
- **THEN** it SHALL include the original `security` and `utility` values
- **THEN** it SHALL identify the source and transformation of any derived attack-success field

### Requirement: Outputs Are Auditable And Joinable
The benchmark SHALL write separate Attention and Shapley JSONL records that share a stable target identity composed from suite, user task, injection task, selected polluted-tool message index, and selected assistant message index. Each row SHALL record the source path, target scope, target kind, target text, selection policy, region spans, and validation status.

#### Scenario: Join Attention and Shapley records
- **WHEN** both attribution methods complete for a target
- **THEN** their records SHALL share the same stable target identity
- **THEN** the identity SHALL resolve to exactly one source trajectory and one selected assistant message

### Requirement: Summaries And Visualizations Preserve Experimental Scope
The benchmark SHALL summarize corpus eligibility, processing failures, source labels, target kinds, Attention metrics, and Shapley contributions without mixing target scopes. It SHALL generate Attention and Shapley visualizations that expose player regions and retain `SPECIAL` separately for Attention diagnostics.

#### Scenario: Summarize the completed benchmark
- **WHEN** benchmark outputs are summarized
- **THEN** the summary SHALL report candidate, eligible, attempted, successful, and failed counts
- **THEN** it SHALL group attribution statistics by `target_scope`, suite, source security label, and target kind
- **THEN** visualizations SHALL identify the first polluted tool response and next assistant target for each displayed case
