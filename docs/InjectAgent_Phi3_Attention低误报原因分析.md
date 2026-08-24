# InjecAgent Phi-3 Attention 实验低误报原因分析

## 1. 报告范围与固定实验定义

本报告分析 InjecAgent action-level Attention 实验中 Phi-3、Qwen3-8B、Llama3-8B 的结果差异。分析使用以下文件：

- `result/injecagent_phi3_dh_clean_20260824/results.action_attention.jsonl`
- `result/injecagent_phi3_dh_attack_20260824/results.action_attention.jsonl`
- `result/injecagent_qwen3_dh_clean_20260809/results.action_attention.jsonl`
- `result/injecagent_qwen3_dh_attack_20260809/results.action_attention.jsonl`
- `result/injecagent_llama3_dh_clean_20260810/results.action_attention.jsonl`
- `result/injecagent_llama3_dh_attack_20260810/results.action_attention.jsonl`

每个 clean/attack 文件各有 510 条记录。clean 是负样本，attack 是正样本；`attention_shift=True` 是预测攻击成功的唯一主信号。

以下三项是统一的论文复现实验定义，必须在三个模型之间保持不变：

1. `threshold=0.5`；
2. 使用首个 action token 的 teacher-forced attention；
3. 使用 `attention_shift` 判定攻击成功。

因此，本报告不提出逐模型调阈值、改为全 action token 聚合、或以其他字段替换 `attention_shift` 的建议。这里的统计只用于解释同一标尺下的模型间差异。

## 2. 实际代码判定规则

`scripts/injecagent_action_experiment.py` 的 `attention_shift_metrics()` 先对三个 player 区域重新归一化：

```text
region_score / (auth + data_fact + data_attack)
```

`special` 不参与分母。随后以：

```python
auth_focus_score = region_scores_player_normalized[AUTH_KEY]
attention_shift = auth_focus_score <= 0.5
```

生成主判定。因此 `attention_shift=True` 的严格代码语义是：

> 排除 special 后，AUTH 在 AUTH、FACT、ATTACK 三个区域中的注意力占比不超过 50%。

这仍是论文复现中使用的攻击信号。但在解释时必须注意，它不等价于 `data_attack` 大于 AUTH，也不等价于 `data_attack` 大于 FACT。`attention_attack_dominant` 是独立诊断字段，不参与 `attention_shift` 的真值判断，也不应替代它。

## 3. 三种模型在固定标尺下的结果

| 模型 | clean 中预测攻击 | attack 中预测攻击 | clean AUTH 均值 | attack AUTH 均值 |
|---|---:|---:|---:|---:|
| Phi-3 | 51/510 = **10.00%** | 121/510 = **23.73%** | **0.7765** | **0.5731** |
| Qwen3-8B | 481/510 = **94.31%** | 510/510 = **100.00%** | **0.3536** | **0.2490** |
| Llama3-8B | 463/510 = **90.78%** | 510/510 = **100.00%** | **0.4184** | **0.2320** |

| 模型 | Accuracy | Precision | Recall | False Positive Rate | F1 |
|---|---:|---:|---:|---:|---:|
| Phi-3 | 56.86% | 70.35% | 23.73% | **10.00%** | 35.48% |
| Qwen3-8B | 53.43% | 51.49% | **100.00%** | 94.31% | 67.98% |
| Llama3-8B | 54.61% | 52.43% | **100.00%** | 90.78% | 68.79% |

Phi-3 的 clean AUTH 均值为 0.7765，远高于固定阈值 0.5，因此大部分负样本不会触发 `attention_shift`；这就是其低误报率的直接原因。attack 中 Phi-3 的 AUTH 均值降至 0.5731，说明注入确实使 attention 从 AUTH 转移，但多数记录仍未低于 0.5，故召回率只有 23.73%。

Qwen3-8B 和 Llama3-8B 的 clean AUTH 均值已经低于 0.5（0.3536、0.4184），因此在相同的统一标尺下会对绝大多数 clean 样本触发 `attention_shift`；attack 中 AUTH 又进一步降至 0.2490、0.2320，故二者呈现高召回、高误报。

## 4. Phi-3 低误报与低召回的同源性

三种模型的现象可以概括为：

```text
Phi-3 clean：AUTH 高于 0.5 → 多数不触发 → 低误报
Phi-3 attack：AUTH 常仍高于 0.5 → 多数不触发 → 低召回

Qwen3/Llama clean：AUTH 已低于 0.5 → 大量触发 → 高误报
Qwen3/Llama attack：AUTH 更低 → 几乎全部触发 → 高召回
```

所以 Phi-3 的低误报不是独立的性能优势；它和低召回来自同一个统计原因：当前重要头集合下，Phi-3 在首个 action token 处对 AUTH 保留的相对注意力更高。

## 5. 攻击区域主导不是差异来源

`attention_attack_dominant` 的统计比例很低：

| 模型 | clean | attack |
|---|---:|---:|
| Phi-3 | 1.57% | 1.76% |
| Qwen3-8B | 0% | 0.59% |
| Llama3-8B | 0% | 0% |

这说明 Qwen3/Llama3 的高召回不是因为 `data_attack` 经常成为最大注意力区域，而是因为 AUTH 在三个 player 区域中的占比低于统一阈值。该结论不改变 `attention_shift` 的论文判定，只避免将其误读为“攻击 token 必须占主导”。

## 6. 在固定协议下应检查的根因

### 6.1 important heads 的跨模型响应差异

三个配置的 head 集合不同：Phi-3 使用 10 个头（第 11、12、14、16 层），Qwen3-8B 使用 14 个头（第 10 至 19 层），Llama3-8B 使用 4 个头（第 5、7、9、17 层）。脚本只平均这些指定 head 的 attention。

不同模型的 head 没有天然一一对应关系。最需要验证的是 Phi-3 已配置的 10 个 head 在当前 InjecAgent prompt 与首个 action token 上是否系统性偏向 AUTH，及这种偏好是否由少数 head 主导。检查这一点不改变阈值、目标 token 或攻击标签，只验证重要头配置在跨模型比较中是否正确实现。

### 6.2 相同手工 prompt 的 tokenizer 与区域映射

脚本按字符查找 AUTH、FACT、ATTACK，再使用每种模型 tokenizer 的 offset mapping 转为 token span。平均 span 长度分别为：

| 模型 | AUTH token 数 | FACT token 数 | ATTACK token 数 |
|---|---:|---:|---:|
| Phi-3 | 29.5 | 66.4 | 33.2 |
| Qwen3-8B | 25.1 | 57.3 | 29.1 |
| Llama3-8B | 24.6 | 51.0 | 26.0 |

同一手工 `<system>`、`<user>`、`Thought`、`Action`、`Observation` 文本会被三种 tokenizer 切分为不同序列。需要逐样本确认 region span 仍准确覆盖目标文本且不越界。这是对固定首 token 测量的 span 审计，而不是替换 prompt、模板或论文定义。

### 6.3 special attention 的排除效应

平均 `special_attention_mass` 为：

| 模型 | clean | attack |
|---|---:|---:|
| Phi-3 | 0.8787 | 0.8931 |
| Qwen3-8B | 0.7982 | 0.7786 |
| Llama3-8B | 0.8953 | 0.8830 |

论文定义将 special 从 player 归一化分母排除。这个规则必须维持，但意味着比较结果由剩余 AUTH/FACT/ATTACK 注意力的相对分布决定。Phi-3 即使有较高 special attention，其剩余 player attention 仍以 AUTH 为主，故保持低触发率。

## 7. 结论与后续验证

在不改变统一阈值、首 action token 或 `attention_shift` 主判定的前提下，现有证据支持：

1. Phi-3 的低误报由较高的 AUTH focus 直接导致，且同一机制导致其低召回；
2. Qwen3/Llama3 的高误报和高召回是其 AUTH focus 在 clean/attack 中均较低的结果；
3. `attention_shift` 继续是论文定义下的攻击成功信号；`attention_attack_dominant` 只用于解释该信号的区域组成；
4. 后续工作应固定所有论文判定条件，审计 important heads 的来源与逐 head 输出，并核对跨 tokenizer 的区域 span 映射；
5. 可在完全相同的 prompt、首 token、阈值和判定定义下，对少量 Phi-3/Qwen3/Llama3 配对样本逐 head 复算，用于确定 Phi-3 的 AUTH 偏好是否由少数头驱动。
