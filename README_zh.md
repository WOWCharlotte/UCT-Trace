# UTC-Trace

[English](README.md)

UTC-Trace 用于研究工具型大语言模型代理中的**未经授权的控制转移**。它解释一条已观测的助手回复或工具调用，主要得到用户指令、工具结果中的任务事实，还是其中的注入指令支持。论文将方法称为 **UCT-Trace**；本仓库按项目要求使用 **UTC-Trace** 作为名称。

仓库包含 **InjecAgent** 与 **AgentDojo** 的动作级实验、Attention Tracker 对照基线、独立的行为裁判和分析脚本。归因阶段需要访问模型权重与输出 token 概率，仅有模型聊天 API 不足以运行该阶段。

## 方法简介

1. 将上下文分为 `AUTH`（用户授权指令）、`FACT`（工具结果中的任务相关事实）和 `ATTACK`（注入指令），其余上下文保持固定。
2. 固定已观测的目标动作，对三个区域的全部八个联盟进行评估。对排除的区域置零 token embedding，以目标动作的平均 token 对数概率计算精确 Shapley 贡献。
3. 当 `M_attack = φ_ATTACK − max(0, φ_AUTH) − max(0, φ_FACT) > 0` 时，将该动作列为 ATTACK 主导候选。
4. 对候选动作核验攻击者指定的任务是否实际执行。**只有 ATTACK 归因占优且行为执行得到核验，才预测攻击成功。**提及或拒绝注入指令不算执行。

这是针对已发生动作的事后解释流程，不会阻断代理动作。Attention Tracker 作为对照基线保留。

## 目录结构

```text
baselines/attention_tracker/   原 Attention Tracker 模型与检测器代码
configs/model_configs/        模型与注意力头配置
data/injecagent/              InjecAgent 输入与生成的动作记录
data/agentdojo/               AgentDojo 轨迹
scripts/                      归因、行为裁判、分析与绘图
tests/                        单元和流程测试
docs/                         研究记录与辅助资料
result/                       本地输出（已被 Git 忽略）
```

## 快速启动

以下命令均在仓库根目录运行。需要具备 PyTorch 和 Transformers 的 Python 环境；GPU 推理还需要兼容的 CUDA 环境。

```bash
python -m pip install -r requirements.txt
python -m pip install pytest
```

先在不加载模型的情况下检查仓库自带的 InjecAgent 记录：

```bash
python scripts/injecagent_action_experiment.py --validate_only --limit 1
```

运行单样本归因前，将 `configs/model_configs/qwen3_8b-attn_config.json` 中的 `model_id` 改为本地或可访问的 Qwen3-8B 模型：

```bash
python scripts/injecagent_action_experiment.py --model_name qwen3_8b-attn --limit 1
```

结果写入 `result/injecagent_qwen3_dh/`，包括 `results.action_shapley.jsonl` 和 `results.action_attention.jsonl`。无需外部裁判服务即可先审计 ATTACK 主导候选：

```bash
python scripts/llm_judge_attack_behavior.py --mode audit --dataset injecagent --input result/injecagent_qwen3_dh/results.action_shapley.jsonl --output result/judge.audit.jsonl --summary result/judge.audit.summary.json
```

核验攻击任务是否执行时，配置 `JUDGE_API_KEY`，并按需配置兼容 OpenAI 接口的 `JUDGE_MODEL` 和 `JUDGE_BASE_URL`；对同一归因输入使用 `--mode judge`，指定新的输出路径。裁判结果单独写入文件，不覆盖归因记录。

AgentDojo 可先运行 `python scripts/agentdojo_attribution_experiment.py --audit-only`，语料与模型选项见脚本的 `--help`。测试命令为 `python -m pytest tests -q`。

## 许可证与上游材料

UTC-Trace 的原创贡献采用 [MIT 许可证](LICENSE)。仓库包含来自 [khhung-906/Attention-Tracker](https://github.com/khhung-906/Attention-Tracker) 的基线代码；其上游 README 标注 **CC BY-NC 4.0**。这部分材料及第三方数据**不会因本仓库采用 MIT 而重新授权**。复用前请阅读 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
