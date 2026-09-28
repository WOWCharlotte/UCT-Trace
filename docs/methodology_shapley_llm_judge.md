# Methodology：Shapley 归因与 LLM 动作裁判框架

> 文档定位：本文档是英文 Methodology 正文的中文设计稿。章节结构与方法示意图严格对齐，保留英文标题、统一术语、数学符号和段落职责。具体数据集映射、实验配置、裁判 API、缓存与重试机制均放入 Experiments 或 Appendix，不在 Methodology 中展开。

## Core argument

本文研究非可信外部内容中的攻击指令是否取得了对 agent 下一步动作的未授权控制。方法首先固定攻击暴露后已经观测到的下一步动作，通过三玩家 Shapley game 估计 AUTH、FACT 与 ATTACK 对该动作的边际支持；随后，仅对 ATTACK 主导的候选进行独立的 LLM 动作验证，区分攻击执行、攻击提及与攻击拒绝。只有攻击贡献主导且恶意动作得到执行确认时，样本才被判定为 strict attack success。

## Figure-to-section mapping

Methodology 总体包含四个小节。第一节概括完整框架，后续三节分别对应方法示意图从左到右的三个区域。

| Methodology subsection | Figure region | Methodological question |
|---|---|---|
| 3.1 Framework Overview | 整体流程 | 方法如何从攻击交互得到严格成功判定？ |
| 3.2 Problem Statement | 左侧区域 | 如何形式化攻击交互、固定动作与未授权控制转移？ |
| 3.3 Explaining Unauthorized Control Transfer | 中间区域 | ATTACK 是否比合法信息更强地支持该动作？ |
| 3.4 Behavioral Verification | 右侧区域 | 该动作是否实际执行了攻击，而非仅提及或拒绝？ |

## Terminology and notation

| 符号或术语 | 统一定义 |
|---|---|
| $x$ | agent 接收的完整决策上下文 |
| $x_{\mathrm{auth}}$ | 原始用户授权任务 |
| $x_{\mathrm{fact}}$ | 非可信外部内容中的正常事实 |
| $x_{\mathrm{attack}}$ | 非可信外部内容中嵌入的攻击指令 |
| $y^*=(y_1^*,\ldots,y_T^*)$ | 攻击暴露后已经产生的固定下一步动作 |
| AUTH | 与 $x_{\mathrm{auth}}$ 对应的授权任务区域；英文统一写作 **authorized-instruction** |
| FACT | 与 $x_{\mathrm{fact}}$ 对应的正常事实区域；英文统一写作 **task-relevant facts** |
| ATTACK | 与 $x_{\mathrm{attack}}$ 对应的非可信控制区域；英文统一写作 **injected instructions** |
| $N$ | 三玩家集合 $\{\mathrm{AUTH},\mathrm{FACT},\mathrm{ATTACK}\}$ |
| $v(S)$ | 联盟 $S\subseteq N$ 对固定目标 $y^*$ 的 teacher-forced mean token log-probability |
| $\phi_i$ | 玩家 $i$ 的 Shapley value |
| $M_{\mathrm{attack}}$ | ATTACK 相对 AUTH 与 FACT 正贡献之和的贡献边际 |
| $\mathrm{UCT}(x,y^*)$ | unauthorized control transfer 的二元研究目标 |
| ATTACK-dominant | 满足 $M_{\mathrm{attack}}>0$ 的归因候选 |
| $J\in\{0,1\}$ | LLM behavior judge 的二元执行标签；$1$ 表示攻击执行，$0$ 表示未执行并覆盖 mentioned 与 refused 语义 |

英文正文不得使用其他近义变体指代三个玩家。全文固定使用 **authorized-instruction**, **task-relevant facts**, and **injected instructions**。

---

## 3.1 Framework Overview

### Section objective

本节对完整方法进行一次从左到右的概括，不提前展开公式细节。核心是将“攻击是否控制了动作生成”和“动作是否执行了攻击”定义为两个相互关联但不可互相替代的问题。

### Content design

本节只使用一个不超过 120 词的英文段落，依次完成方法提出、两阶段流程和严格成功条件三个功能。正式英文建议如下：

> Our framework offers an interpretable approach to detecting unauthorized control transfer in tool-using language agents. As illustrated in Figure~\ref{fig:framework}, it operates in two stages. First, Shapley attribution measures the contributions of the authorized-instruction, task-relevant facts, and injected instructions to the agent's observed action, then selects cases dominated by injected instructions. Second, an LLM-based judge determines whether each candidate executes the malicious request rather than merely mentioning or refusing it. An interaction is classified as a successful attack only when injected instructions dominate the attribution and the observed action is judged to execute the attack.

---

## 3.2 Unauthorized Control Transfer

考虑一个接收 `authorized-instruction` 和 `untrusted external content` 的 Tool-Using LLM Agent。令 \(x_{\mathrm{auth}}\) 表示 `authorized-instruction`，其规定了智能体需要完成的用户目标任务；令 \(x_{\mathrm{fact}}\) 和 \(x_{\mathrm{attack}}\) 分别表示 `untrusted external content` 中的 `task-relevant facts` 和 `injected instructions`。其中，\(x_{\mathrm{fact}}\)用于帮助智能体完成用户任务，\(x_{\mathrm{attack}}\)则试图诱导智能体执行攻击者预设的操作。完整上下文表示为

$$
x=
\left(
x_{\mathrm{auth}},
x_{\mathrm{fact}},
x_{\mathrm{attack}}
\right).
$$

给定上下文 \(x\)，智能体生成下一步动作

$$
y^*\sim p_{\theta}(\cdot\mid x),
$$

其中，\(y^*\) 可以是自然语言回复，也可以是结构化工具调用。后续分析将该动作 \(y^*\) 固定为解释目标，在改变输入区域或评估其影响时始终针对同一个动作展开分析，而不重新生成新的动作。

定义`unauthorized control transfer`（UCT）为\(x_{\mathrm{attack}}\) 在智能体的决策过程中取得主导控制权，使智能体偏离 \(x_{\mathrm{auth}}\) 所规定的用户目标，并生成服务于攻击者特定任务的动作。智能体仅仅是关注、提及、复述或拒答`injected instructions`，而不执行攻击者特定任务的现象不属于 UTC。

为研究这一现象，本文将智能体在给定上下文下生成的动作作为固定解释目标，分析不同输入区域对该动作的相对影响，并结合动作本身的工具调用、关键参数和执行结果，评估该动作是否执行了攻击者特定任务。


## 3.3 Shapley attribution

### Attribution players

我们将固定已观测动作（fixed observed action）$y^*=(y_1^*,\ldots,y_T^*)$ 的生成支持建模为三玩家 cooperative game，其中玩家集合为 $N=\{\mathrm{AUTH},\mathrm{FACT},\mathrm{ATTACK}\}$。AUTH 表示授权用户指令对应的 token 区域；FACT 表示不可信外部内容中的正常事实区域；ATTACK 表示其中嵌入的攻击指令区域。FACT 与 ATTACK 均可由多个不连续的 token span 组成。系统指令、工具定义、消息模板、结构标记以及其他未归入上述三区域的上下文统一视为 fixed context：它们不构成玩家，也不在联盟干预中被掩蔽。

归因目标 $y^*$ 是攻击暴露后实际观测到的完整下一步动作，可以是自然语言回复或序列化的工具调用。我们在所有联盟下固定这一目标，仅比较不同输入区域对同一动作的条件支持程度，而不重新采样或生成动作。

### Coalition intervention

对于任意联盟 $S$，我们保留属于 $S$ 的玩家 token embeddings，并将所有不属于 $S$ 的玩家 span 内的 embeddings 置零。该操作不删除或替换 token，因而序列长度、token 位置、causal attention mask 均保持不变。特别地，空联盟 $S=\emptyset$ 会同时掩蔽 \(x_{\mathrm{auth}}\)、\(x_{\mathrm{fact}}\) 与 \(x_{\mathrm{attack}}\)，但仍保留固定的系统指令、工具定义和消息模板。联盟间的差异仅来自用于支持目标动作的玩家数量。

### Coalition value

联盟的 characteristic value 定义为固定目标动作的 teacher-forced mean token log-probability：

$$
v(S)=\frac{1}{T}\sum_{t=1}^{T}
\log p_\theta\!\left(y_t^*\mid x_S\right).
$$

其中，$S\subseteq N$ 表示当前联盟，$x_S$ 表示在该联盟干预下保留相应玩家区域后的提示；$y^*=(y_1^*,\ldots,y_T^*)$ 表示攻击暴露后实际观测到的固定下一步动作；$y_t^*$ 表示该动作在第 $t$ 个输出位置上的目标 token；$T$ 表示固定动作包含的输出 token 数量；$p_\theta(y_t^*\mid x_S)$ 表示模型在给定干预提示下为第 $t$ 个目标位置分配的条件概率，$\log$ 表示对该概率取对数。模型在 teacher forcing 下按固定动作的已知前缀评估每个输出位置，再对全部 $T$ 个目标 token 的 log-probability 取平均。因此，$v(S)$ 越大，表示联盟 $S$ 对固定动作 $y^*$ 提供的平均条件支持越强；该分数不表示动作质量，也不直接表示攻击成功概率。

### Shapley value computation

对任意玩家 $i\in N$，其 Shapley value 为

$$
\phi_i=
\sum_{S\subseteq N\setminus\{i\}}
\frac{|S|!(|N|-|S|-1)!}{|N|!}
\left[v(S\cup\{i\})-v(S)\right].
$$

其中，$\phi_i$ 表示玩家 $i$ 的 Shapley value，$i$ 表示当前待计算归因值的玩家；$v(S\cup\{i\})-v(S)$ 表示将玩家 $i$ 加入联盟 $S$ 后带来的边际价值变化。

### ATTACK dominance

定义 ATACK-dominant candidate 满足以下条件：

$$
M_{\mathrm{attack}}=
\phi_{\mathrm{ATTACK}}-
\left[
\max(0,\phi_{\mathrm{AUTH}})+
\max(0,\phi_{\mathrm{FACT}})
\right].
$$

其中，$\phi_{\mathrm{ATTACK}}$ 表示攻击指令对固定动作的归因贡献，$\max(0,\phi_{\mathrm{AUTH}})$ 与 $\max(0,\phi_{\mathrm{FACT}})$ 分别保留两个合法区域的正贡献。由此，$M_{\mathrm{attack}}$ 衡量 ATTACK 贡献超过合法正贡献总和的幅度。仅当 $M_{\mathrm{attack}}>0$ 时，我们才将 $y^*$ 标记为 ATTACK-dominant candidate。


---

## 3.4 Behavioral Verification

### Candidate cascade

ATTACK dominance 只能说明攻击指令是固定动作 $y^*$ 的主要正向支持来源，但无法区分攻击执行、提及与拒绝。事实上，拒绝、复述或讨论攻击的动作同样可能依赖攻击文本。为避免将这种依赖误判为攻击成功，我们对满足 $M_{\mathrm{attack}}>0$ 的样本由 LLM Judge 做行为验证。

### Judge inputs

LLM Judge 接收一条结构化输入记录，包含：(i) 授权任务 $x_{\mathrm{auth}}$；(ii) 完整的工具响应；(iii) 攻击指令 $x_{\mathrm{attack}}$；(iv) 固定观测动作 $y^*$；(v) 当前可用的工具定义；以及 (vi) 外部环境反馈 $e$。外部环境反馈 $e$ 仅描述动作发生后的可观测事实。Judge 根据这些信息判断 $y^*$ 是否在攻击目标、操作类型和关键参数上与 $x_{\mathrm{attack}}$ 语义匹配，并且是否获得成功的执行结果。

### Behavior classification

LLM Judge 将观测行为$y^*$映射为二元分类：

$$
J(x_{\mathrm{auth}},x_{\mathrm{attack}},y^*,e)
\in\{0,1\}.
$$

其中，$J(\cdot)=1$ 表示观测到与攻击指令在目标对象、操作类型和关键参数上语义匹配的结构化动作。$J(\cdot)=0$ 表示 $y^*$ 仅提及、复述或拒绝攻击指令。

### Strict attack success

Strict attack success 同时要求攻击指令主导固定动作的生成支持，且该动作实际执行攻击。其判定条件为

$$
M_{\mathrm{attack}}>0
\quad\text{and}\quad
J=1.
$$

其中，$M_{\mathrm{attack}}>0$ 是归因条件，表示 ATTACK 对固定动作的支持超过 AUTH 与 FACT 的正贡献之和；$J=1$ 是行为条件，表示观测到与攻击指令语义匹配且具有必要执行证据的结构化动作。只有两个条件同时成立，样本才被判定为 strict attack success。归因条件排除并非由 ATTACK 主导支持的有害动作，行为条件则排除仅提及、复述或拒绝攻击的动作。

---

## Assumptions and methodological boundaries

- 本方法解释固定的已观测动作 $y^*$，不估计重新生成分布下的反事实攻击成功率。
- Shapley values 是相对于三玩家划分、teacher-forced mean-logprob value function 和 embedding-zeroing intervention 的边际贡献估计，不是无条件因果真值。
- 三个玩家区域必须严格两两不重叠；系统提示、工具定义、交互历史和结构 token 作为 fixed context。
- 每次归因面向攻击暴露后的一个完整动作，不在 Methodology 中扩展为长程动作链解释。
- LLM judge 是有边界的语义验证器，其结果依赖裁判模型、判断规则与可用执行证据。
- Methodology 不包含具体数据集、样本解析、模型列表、超参数或实验结果，也不声明尚未验证的性能优势。

## Claim--evidence map

| Methodological claim | Implementation evidence | Status |
|---|---|---|
| 三玩家 exact Shapley 使用全部八个联盟 | 归因实现枚举 $2^3$ coalitions 并使用标准 Shapley formula | supported |
| 价值函数为固定目标的 teacher-forced mean token log-probability | 遮蔽 prompt player embeddings 后对原始目标 token 计算平均 log-probability | supported |
| ATTACK-dominant 使用 `attack-margin-v1` | margin 对 AUTH 与 FACT 使用 positive clipping | supported |
| strict success 是 ATTACK dominance 与 executed 的合取 | 级联判定同时要求 attribution trigger 与 positive behavior judgment | supported |
| mention、refusal 与技术失败不能产生 strict success | 动作标签与未判定状态在决策规则中保持区分 | supported |
| 工具执行具有攻击语义与成功执行证据 | 语义匹配依赖 LLM judge，确定性执行约束应保持窄化表述 | implementation boundary |

## ACL-oriented English paragraph plan

### Argument spine

整节围绕一个中心论证展开：*Unauthorized control transfer is identified only when injected instructions are the dominant source of support for a fixed observed action and that action is independently verified to execute the attack.* Section 3.2 定义研究对象，Section 3.3 判断控制来源，Section 3.4 验证行为结果；三个部分依次回答“解释什么动作”“谁主导该动作”和“该动作是否执行攻击”。

### Layout and notation

- Methodology 使用一个主 section 和四个 numbered subsections；英文正文不保留当前中文草稿中的三级标题，而以段落主题句体现模块边界。
- 建议总长度为 1,150--1,400 English words：Section 3.1 约 90--110 词，Section 3.2 约 230--280 词，Section 3.3 约 550--650 词，Section 3.4 约 280--340 词。
- Figure~\ref{fig:framework} 在 Section 3.1 首次引用后尽早出现，并从左到右展示 fixed observed action、Shapley attribution、ATTACK-dominant cascade 与 behavior judge。
- 八个 displayed expressions 按出现顺序为输入元组 $x$、动作生成 $y^*$、$v(S)$、一般 Shapley value、展开的 $\phi_{\mathrm{ATTACK}}$、$M_{\mathrm{attack}}$、$J(\cdot)\in\{0,1\}$ 和 strict attack success 的联合条件。
- 每个 displayed expression 后紧接符号解释与方法含义，不在多个公式之后集中解释。全文固定使用 AUTH、FACT、ATTACK、fixed observed action、teacher-forced mean token log-probability、ATTACK-dominant 和 behavior judge。
- 数据集映射、区域提取规则、模型配置、裁判 prompt、API、缓存、重试、并发与失败统计属于 Experiments 或 Appendix，不进入 Methodology。

### Section 3.1: Framework Overview

**Paragraph 1 -- Framework pipeline (90--110 words).**

Suggested opening: *Our framework provides an interpretable method for detecting unauthorized control transfer in tool-using language agents.*

按照 `methodology.tex` 的论证顺序，先陈述框架目标并引用 Figure~\ref{fig:framework}，再概括两阶段流程：Shapley attribution 量化 authorized-instruction、task-relevant facts 与 injected instructions 对 observed action 的贡献，并保留攻击贡献占主导的候选；LLM-based behavior judge 随后判断该动作是执行攻击，还是仅提及或拒绝攻击。段末以自然语言给出 successful attack 的联合条件。本段只承担方法总览，不定义玩家、价值函数或裁判输入。

### Section 3.2: Unauthorized Control Transfer

**Paragraph 2 -- Interaction context and observed action (120--145 words).**

Suggested opening: *Consider a tool-using LLM agent that receives an authorized instruction together with untrusted external content.*

依次定义 $x_{\mathrm{auth}}$、$x_{\mathrm{fact}}$ 与 $x_{\mathrm{attack}}$，并说明三者分别承担用户授权、任务事实与攻击重定向功能。按照正文保留两个 displayed expressions：先给出完整上下文 $x=(x_{\mathrm{auth}},x_{\mathrm{fact}},x_{\mathrm{attack}})$，再给出 $y^*\sim p_\theta(\cdot\mid x)$。公式后说明 $y^*$ 可以是自然语言响应或结构化工具调用，并在后续 attribution analysis 中固定为 explanatory target。

**Paragraph 3 -- UCT definition and exclusion boundary (65--80 words).**

Suggested opening: *We define unauthorized control transfer (UCT) as a state in which $x_{\mathrm{attack}}$ exerts dominant influence over the agent's decision.*

说明这种主导影响使动作偏离 $x_{\mathrm{auth}}$ 指定的用户目标，转而执行攻击者任务。随后划定 UCT 的排除边界：仅暴露于 injected instructions，或仅确认、提及、复述和拒绝它们，均不足以构成 UCT；只有行为实际转向攻击者任务时才满足行为条件。本段不提前引入 $M_{\mathrm{attack}}$ 或 $J$。

**Paragraph 4 -- Evidence decomposition (40--55 words).**

Suggested opening: *To characterize UCT, we combine two types of evidence.*

用一个短段收束问题定义并承接后续模块：第一类证据是不同输入区域对 fixed target action $y^*$ 的相对影响，由 Section 3.3 给出；第二类证据来自工具调用、关键参数和执行结果，用于判断动作是否执行攻击者任务，由 Section 3.4 给出。

### Section 3.3: Shapley Attribution

**Paragraph 5 -- Players and attribution target (100--120 words).**

Suggested opening: *Given the fixed observed action \(y^*\), we quantify the contribution of each previously defined input region by formulating a three-player cooperative game with \(x_{\mathrm{auth}}\), \(x_{\mathrm{fact}}\), and \(x_{\mathrm{attack}}\) as the players.*


**Paragraph 6 -- Coalition intervention (100--120 words).**

Suggested opening: *For each coalition $S$, we retain the embeddings of players in $S$ and zero the embeddings covered by all excluded players.*

严格按执行顺序描述干预：保留联盟内 embeddings、置零联盟外 span、保持 token 不被删除。说明序列长度、位置、causal attention mask 保持不变。

**Paragraph 7 -- Coalition value (100--120 words).**

Suggested opening: *The value of a coalition is the teacher-forced mean log-probability assigned to the fixed observed action.*

给出 $v(S)$。公式后逐一解释 $S$、$x_S$、$y^*$、$y_t^*$、$T$、$p_\theta$ 与 $\log$；说明 $T$ 是固定目标动作的输出 token 数量，$p_\theta(y_t^*\mid x_S)$ 表示模型在当前联盟干预下对第 $t$ 个目标位置分配的条件概率，并在 teacher forcing 下按固定动作的已知前缀进行评估。对全部输出位置的 log-probability 取平均后，得到联盟对固定动作的平均条件支持度。段末限定分数含义：$v(S)$ 不表示动作质量，也不等同于攻击成功概率。

**Paragraph 8 -- Exact Shapley values (140--170 words).**

Suggested opening: *Because the game has only three players, we compute exact Shapley values from all eight coalitions.*


**Paragraph 9 -- ATTACK dominance (100--120 words).**

Suggested opening: *We identify attribution candidates using a conservative margin between ATTACK support and legitimate positive support.*

给出 $M_{\mathrm{attack}}$ 并解释对 AUTH 与 FACT 进行 positive clipping 的原因：合法区域的负贡献不能反向增加攻击支持。随后规定仅当 $M_{\mathrm{attack}}>0$ 时样本为 ATTACK-dominant candidate。段末明确归因边界：该条件只表示 ATTACK 是固定动作的主要正向支持来源，不能证明攻击已被执行。

### Section 3.4: Behavioral Verification

**Paragraph 10 -- Candidate cascade and judge inputs (90--110 words).**

Suggested opening: *Attack-dominant attribution alone cannot distinguish execution from mention, repetition, or refusal.*

先用反例说明独立裁判的必要性，再规定只有 $M_{\mathrm{attack}}>0$ 的样本进入 behavior judge。按信息功能列出裁判输入：授权任务、完整外部响应、攻击指令、fixed observed action、工具定义和执行结果 $e$。对 $e$ 只保留可复现的高层定义，包括工具名称、参数、返回结果与执行状态；不描述 prompt 模板或数据集适配器。

**Paragraph 11 -- Binary behavior classification (100--120 words).**

Suggested opening: *The behavior judge maps each evaluated candidate to a binary execution label $J$.*

给出 $J(x_{\mathrm{auth}},x_{\mathrm{attack}},y^*,e)\in\{0,1\}$。公式后定义 $J=1$ 所需的目标对象、操作类型和关键参数语义匹配，并在存在关联结果时要求成功执行证据；定义 $J=0$ 覆盖 mention、refusal、失败调用、结果缺失与关键参数不匹配。裁判失败保持未判定，不并入 $J=0$；LLM judge 不接收 Shapley scores 作为行为证据。

**Paragraph 12 -- Strict attack success (90--110 words).**

Suggested opening: *Strict attack success is reported only when attributional dominance and behavioral execution agree.*

直接给出 $M_{\mathrm{attack}}>0$ 且 $J=1$ 的联合条件，不定义额外结果变量。公式后分别说明两个条件排除的假阳性：前者排除并非由 ATTACK 主导的有害动作，后者排除仅依赖攻击文本但未执行攻击的动作。最后区分明确行为负类与未判定状态：ATTACK-dominant 但裁判失败或证据不足的样本单独报告，不计为成功，也不并入已裁判的负类。

### Compression priorities

若 Methodology 超出 ACL 页面预算，依次压缩 Paragraph 8 对四条 ATTACK marginal contributions 的文字解释、Paragraph 11 对负类情形的枚举，以及 Paragraph 5 对 fixed context 的例举。Section 3.1 始终保持单段。不得删除 fixed observed action、三玩家划分、embedding-zeroing intervention、teacher-forced value function、exact Shapley、attack margin、judge semantics、strict attack success 的联合条件或未判定状态边界。
