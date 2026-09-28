# InjectAgent-Phi3 Attention 头选择实验结果

## 实验目的

本实验用于验证 Phi-3 中 `important_heads` 的层位选择是否会显著影响 Attention 实验结果。三组配置保持重要注意力头数量一致，仅改变所选头位于模型的中后层、后层或最终层附近：

- `phi3-attn-middle`
- `phi3-attn-late`
- `phi3-attn-final`

每组均使用相同的 510 条 clean 样本（负类）和 510 条 attack 样本（正类）。按照论文原始实验约定，使用第一个 action token 的 Attention 结果，并以 `attention_shift=True` 作为“预测攻击成功”的信号。阈值保持统一，不对不同配置进行调节。

## 统计方法

设 attack 为正类、clean 为负类：

- `TP`：attack 且 `attention_shift=True`
- `FN`：attack 且 `attention_shift=False`
- `FP`：clean 且 `attention_shift=True`
- `TN`：clean 且 `attention_shift=False`

指标定义如下：

\[
\text{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}
\]

\[
\text{Precision}=\frac{TP}{TP+FP},\quad
\text{Recall}=\frac{TP}{TP+FN},\quad
\text{FPR}=\frac{FP}{FP+TN}
\]

\[
F1=\frac{2\cdot\text{Precision}\cdot\text{Recall}}
{\text{Precision}+\text{Recall}}
\]

## 实验结果

| 设置 | TP | FP | TN | FN | 准确率 | 精确率 | 召回率 | 误报率 | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `middle` | 507 | 181 | 329 | 3 | 81.96% | 73.69% | 99.41% | 35.49% | 84.64% |
| `late` | 509 | 510 | 0 | 1 | 49.90% | 49.95% | 99.80% | 100.00% | 66.58% |
| `final` | 473 | 481 | 29 | 37 | 49.22% | 49.58% | 92.75% | 94.31% | 64.62% |

对应结果文件为：

- `result/injecagent_phi3_middle_dh_clean_20260809/results.action_attention.jsonl`
- `result/injecagent_phi3_middle_dh_attack_20260809/results.action_attention.jsonl`
- `result/injecagent_phi3_late_dh_clean_20260809/results.action_attention.jsonl`
- `result/injecagent_phi3_late_dh_attack_20260809/results.action_attention.jsonl`
- `result/injecagent_phi3_final_dh_clean_20260809/results.action_attention.jsonl`
- `result/injecagent_phi3_final_dh_attack_20260809/results.action_attention.jsonl`

结果文件均包含 510 条 clean 和 510 条 attack 记录。

## 结果分析

### 1. 注意力头的层位选择明显影响检测结果

三种配置的重要头数量相同、判定阈值相同、输入数据相同，但指标差异很大：

- `middle` 的准确率最高（81.96%），F1 最高（84.64%），同时召回率达到 99.41%。
- `late` 的召回率最高（99.80%），但 510 条 clean 样本全部被判为攻击，误报率为 100%。
- `final` 仍保持较高召回率（92.75%），但误报率为 94.31%，准确率仅 49.22%。

因此，Attention 检测性能并不只由重要头的数量决定，重要头所在的网络层位同样是关键因素。

### 2. `middle` 配置取得了更好的攻防平衡

`middle` 配置在 attack 样本上只漏检 3 条，但在 clean 样本上保留了 329 条真阴性，仅产生 181 条误报。相比之下，`late` 和 `final` 对 clean 输入产生了大规模 `attention_shift`，导致分类器几乎退化为“始终预测攻击”：

- `late`：`TP=509`、`FP=510`，只有 1 条 attack 漏检，没有任何 clean 样本被正确识别为非攻击。
- `final`：`TP=473`、`FP=481`，误报数量远高于真阴性（`TN=29`）。

这说明中后层头可能更能区分攻击导致的工具调用变化与正常 action 生成，而过于靠后的头可能对 action/token 局部模式普遍敏感，从而在 clean 和 attack 上都触发 `attention_shift`。

### 3. 不能只用召回率评价配置

三组配置的召回率都较高，单独看召回率会认为它们都有效；但 `late` 和 `final` 的高召回是以极高误报为代价获得的。精确率、误报率和真阴性数量揭示了这一点：`middle` 的精确率为 73.69%，而 `late` 和 `final` 分别只有 49.95% 和 49.58%。对于实际防御系统，`middle` 更具可用性，因为它在保持高召回的同时显著减少了对正常 action 的错误拦截。

## 基线对照

结果目录中另有未带层位名称的 Phi-3 基线结果：

| 设置 | TP | FP | TN | FN | 准确率 | 精确率 | 召回率 | 误报率 | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Phi-3 基线 | 121 | 51 | 459 | 389 | 56.86% | 70.35% | 23.73% | 10.00% | 35.48% |

基线误报率较低，但召回率只有 23.73%，说明它更保守，漏掉了大量攻击。引入并选择合适的中后层重要头后，`middle` 将召回率提升到 99.41%，同时误报率为 35.49%，在本实验三种层位配置中取得最佳综合结果。

## 结论与注意事项

1. 在统一阈值和统一判定信号下，重要注意力头的层位选择会显著改变 Attention 检测器的决策分布。
2. `phi3-attn-middle` 是本批实验中最合适的配置：高召回、较低误报，并拥有最高准确率和 F1。
3. `late` 和 `final` 的结果表明，选择最终层附近的头可能造成 clean 样本上的系统性 `attention_shift`，不适合作为当前统一标尺下的直接配置。
4. 以上结论是针对当前 Phi-3 模型、数据集、action-token 位置和固定阈值的实验结论；若改变阈值、action-token 位置或数据分布，不能直接假设层位排序保持不变。
