# AgentDojo Attention 与 Shapley 实验

## 实验范围

实验入口为 `scripts/agentdojo_attribution_experiment.py`，只读取：

```text
data/agentdojo/runs/qwen3-8b/*/user_task_*/important_instructions/*.json
```

canonical 数据快照包含 629 条候选轨迹。只有 tool 返回中实际出现完整 `<INFORMATION>...</INFORMATION>` 攻击块的轨迹才进入归因实验：565 条有效，64 条以 `no_information_attack_block` 排除。默认运行会断言这些计数；数据集有意变更时可使用 `--allow-corpus-drift`，但该运行不再与 canonical 结果严格可比。

每条有效轨迹只产生一个解释目标：选择首个包含攻击块的 tool 返回，以及它之后的下一条 assistant 消息。之后的污染 tool 返回和 assistant 消息均不参与本实验。

## 玩家与目标

三个 Shapley 玩家固定为：

- `AUTH`：原始 user prompt，不包含 system prompt。
- `FACT`：首个污染 tool 返回排除所有攻击块后的内容，可以是多个不连续 spans。
- `ATTACK`：首个污染 tool 返回中的所有完整 `<INFORMATION>...</INFORMATION>` spans。

system prompt、chat template、工具调用格式、历史 assistant 消息、其他 tool 返回及未归属 token 都是固定 `SPECIAL` 上下文。`SPECIAL` 只用于 Attention 诊断，不是 Shapley 玩家。

目标 scope 固定为 `next_assistant_message`。AgentDojo 的 `{function, args, id}` tool call 会无损转换为 Qwen chat template 接受的结构，再用本地 Qwen3-8B tokenizer 产生原生 assistant 序列。目标类型记录为 `tool_calls`、`content` 或 `mixed`。

## 归因协议

Attention 对原始 assistant 目标做 teacher forcing。预测每个目标 token 时，从配置的 important heads 取得指向 prompt token 的注意力，先在 heads、再在所有目标 token 上求均值。输出同时包含：

- `region_scores_raw`；
- 包含 `SPECIAL` 的 `region_scores_prompt_normalized`；
- 只以 `AUTH + FACT + ATTACK` 为分母的 `region_scores_player_normalized`；
- `auth_focus_score`、`attention_shift` 和独立的 `attention_attack_dominant`。

Shapley 精确计算三玩家的全部八个联盟。每个联盟都 teacher-force 同一条原始 assistant 目标，价值函数为目标 token 的平均 log probability。排除玩家时只把对应 prompt token embeddings 置零，token 数、位置、attention mask、固定上下文和目标 embeddings 均保持不变。

AgentDojo 原始 `security` 和 `utility` 字段原样保存。便利字段 `attack_success` 定义为 `not security`，并用 `attack_success_source=logical_not_of_agentdojo_security` 声明来源；归因指标不参与标签生成。

## 运行命令

只审计语料，不加载模型：

```bash
python scripts/agentdojo_attribution_experiment.py \
  --audit-only \
  --output-dir result/agentdojo_qwen3_important_instructions
```

运行 canonical 全量 Attention 与 Shapley：

```bash
python scripts/agentdojo_attribution_experiment.py \
  --model-config configs/model_configs/qwen3_8b-attn_config.json \
  --output-dir result/agentdojo_qwen3_important_instructions
```

先用一条真实样本验证：

```bash
python scripts/agentdojo_attribution_experiment.py \
  --limit 1 \
  --output-dir result/agentdojo_qwen3_smoke
```

可以用 `--target-id` 精确选择稳定 target ID，或用 `--skip-attention`、`--skip-shapley` 单独运行一种方法。两种方法不能同时跳过。

## 输出

输出目录包含：

```text
audit.jsonl
audit.summary.json
results.attention.jsonl
results.shapley.jsonl
results.summary.json
attribution_gallery.html
```

Attention 与 Shapley 记录通过 `target_id` 一对一连接。每行保留 source path、消息索引、原始标签、完整目标、玩家文本、token spans、验证结果和方法元数据。summary 按 target scope、suite、security 和 target kind 分组，不混合其他解释 scope。

也可以在实验结束后独立重建可视化：

```bash
python scripts/visualize_agentdojo_attribution.py \
  --shapley result/agentdojo_qwen3_important_instructions/results.shapley.jsonl \
  --attention result/agentdojo_qwen3_important_instructions/results.attention.jsonl \
  --all \
  --output result/agentdojo_qwen3_important_instructions/attribution_gallery.html
```

脚本支持 `--target-id`、`--index`、`--all` 和 `--limit`。页面可按 suite、security 和 target kind 筛选，并可独立显示或隐藏 `AUTH`、`FACT`、`ATTACK`、`SPECIAL` token。

## 验证与局限

单元与回归测试：

```bash
pytest -q \
  tests/test_agentdojo_attribution_experiment.py \
  tests/test_injecagent_action_experiment.py \
  tests/test_shapley.py \
  tests/test_visualize_attention_tokens.py \
  tests/test_visualize_injecagent_actions.py
```

Qwen3-8B 需要约 16 GB 权重显存，完整目标 Attention 还会产生 eager attention tensors；建议一次处理一个 target，并按需要跳过其中一种方法。实验不重跑 AgentDojo、不执行工具、不解释最终环境状态，也不把其他 tool 返回中的事实计入 `FACT`。因此结论只适用于“原始 user prompt、首个污染 tool 返回中的正常内容、同一返回中的攻击块”这三个玩家对下一条 assistant 消息的归因。
