# Dream-RSI: Recursive Self-Improvement through Evolving Worlds (Google / UMD / DeepMind / UVA, 2026-09)

- 作者: Tong Zheng, Xidong Wu, Zheng Zhang 等(1Google, 2UMD, 3Google DeepMind, 4UVA;通讯: xidongwu@google.com, zzhangx@google.com)
- 出处: 项目页 https://dream-rsi.com/assets/dream-rsi.pdf(非 arxiv,36 页),代码 github.com/zhengkid/Dream-RSI
- 性质: agentic 科学发现系统论文 —— 把"递归自我改进(RSI)"作用在**探索策略(meta 层)**上,核心机制是把历史发现记录变成可交互的 replay simulator(world model 的廉价替代),实现离线的策略"做梦"改进。
- 原文缓存: `~/.cache/nanochat/knowledge/pdf/dream-rsi/`(pdf + txt,read-arxiv-paper skill 的 pdf 路由产出)

## 要解决的问题

LLM 驱动的科学/算法发现(AlphaEvolve、SimpleTES 等)本质是"发现循环":生成候选 → 评估 → 反馈 → 改进,常需数千个 proposal–evaluation 周期。**瓶颈在探索策略本身**:

1. 固定策略(人工设计、全程不变)不吸收经验,反复把算力浪费在无效方向;
2. 在线优化探索策略(meta-level)反馈**延迟且昂贵**——评估一个策略要看它 shaping 的整段长程 rollout;且 meta 策略空间巨大,候选策略多,试错成本高。

## 核心 insight:历史发现记录 = 免费的 replay simulator

类比 model-based RL / World Models(Dreamer 家族;Ha & Schmidhuber 2018, Hafner et al. 2019-2025)。**一次完成的发现过程本身就记录了一棵结构化的"发现树"**:探索决策点、每分支的代码、执行结果、评分。**评估替代策略 = 在树上重放**(选不同分支子集/顺序/并行分组/停止时机),只需 reveal 已存节点结果,**零执行成本**,无需重跑 coding agent 或 evaluator。一次昂贵在线运行 → 数千次 off-policy 评估,把 meta-policy 改进从在线试错变成模拟"做梦"。

与已有历史复用范式的区别:不是静态文本上下文(prompt 注入),不是微调训练数据,而是**可交互模拟环境**(见 §5.1:prompt 级语义指导反而有害)。

## Dream-RSI 框架(§3)

轻量编排层使探索显式、可编程(控制分支/并行/停止),**底层 coding agent 不动**。外层迭代 t=1,2,…,三阶段循环:

1. **Online Explore**:策略 π_t 指导 discovery agent 扩展发现树 T_t,追加历史 H_t;
2. **Construct Replay Simulator**:树 → 可复用 simulator pool;
3. **Dreaming-based Policy Improvement**:policy-development agent(LLM)查看 replay 轨迹与得分,**直接改写可执行探索策略代码**,产生 M 个版本,全部在历史树上 replay 评分,选最优为 π_{t+1} 重新上线,闭环。

**形式化要点**:

- 原子操作只有 `CONTINUE(v)`:恢复节点 v 工作区 → 生成并评估一个新子节点(记录 workspace 快照、artifact、诊断、score)。合格集合 A(T)={根}∪{叶子};每轮决策选 batch C,|C|≤W(W=并行 worker 数);选空 batch 终止。
- 在线 rollout:≤K1 轮,转移随机(agent 生成不确定);离线 replay:≤K2 轮,转移确定性(揭示已记录子节点),策略只能看已揭示 prefix(防泄漏,等价于 frozen 的 branch×attempt 网格)。
- **Replay 目标**:V = max_v s_v(发现质量) − β₁·N(执行代价) + β₂·N/轮数(并行奖励),β₁,β₂ 固定系数。
- **单调性保证**:候选集含当前策略,故 π_{t+1} 的历史 replay 得分 ≥ π_t。
- prompt 里还设计了 cross-cycle 的默认 beta 自适应规则(平台期→加大探索 0.1~0.2,证据不足→默认 0.6),以及 `plan_grid`(下一轮 width/depth 规划)。

## 实验(8 任务、3 领域;agent 用 Gemini-3.1 Pro / 3.7-Flash,经 Gemini CLI)

对照基线 **Recursive Fixed Exploration**:同 agent/初始化/首轮预算,策略固定不动;Dream-RSI 从第 2 轮起用 dreaming 改进策略。

1. **算法工程 — Lasso 正则化路径(SimpleTES benchmark,17 合成实例 + 6 held-out)**
   - Gemini-3.1 Pro:仅 **317 次** discovery-agent 调用(固定探索 550,SimpleTES 51,200,~162× 省),平均下游 runtime 3587.1→2931.0ms,6 数据集全面超 sklearn/glmnet;
   - Gemini-3.7-Flash:1879 次调用 vs 3200,runtime 2516.7→2350.6ms;
   - 发现的 solver(附录 C 完整代码):strong-rule screening + **自适应 Cauchy–Schwarz KKT pruning**(界不能证明才算精确梯度,无效则 full refresh)、disjoint active-set(O(1) swap-delete)、lazy Gram、4x 寄存器分块 SIMD + OpenMP、64B 对齐、cache-blocked 融合转置预计算。比 SimpleTES 的"LARS/CD 按维度切换"更原生(自适应性内置于 active-set 优化)。
2. **数学优化(Sum-Diff / Autocorrelation / Circle Packing,10 轮, Gemini-3.1 Pro)**
   - Sum-Diff 1.145427 > SimpleTES 1.143975 > 固定探索 1.144047;Circle Packing 2.635983 追平 SOTA;Autocorrelation 1.456375 有竞争力(SimpleTES 该项更强,但用 51,200 代 vs 本文 <1000 代,>50× 预算差)。
3. **GPU kernel 工程(KernelBench: VGG16 / LayerNorm / ConvDiv / ConvMax,性能=1/ms,正确性校验)**
   - VGG16 / LayerNorm:**2.43× / 1.79× 更少 generations** 达同等性能;
   - ConvDiv / ConvMax:同等预算下性能高 **2.09× / 1.44×**。

## 进一步分析(§5)

1. **历史当语义指导注入 prompt 反而有害**:固定探索与 Dream-RSI 上都变差 —— 长程多分支发现中,强语义归纳偏置过度约束搜索、抑制多样性。可交互 replay > prompt guidance。
2. **探索行为自适应进化**(ConvDiv):性能上升期省算力(每轮评估尝试 110→50),平台期又加力(50→92→110→87→80→91→86→50),与收益曲线同步 —— 策略学会了"该省省、该花花"。

## 相关 work 定位

- 发现系统:AlphaEvolve/OpenEvolve/CodeEvolve/DeltaEvolve 优化候选解;EvoX/SkyDiscover/SwarmResearch 开始优化搜索策略,但 meta 监督贵且延迟 → Dream-RSI 让 meta 优化**递归化 + off-policy**。
- 自进化 agent:多在 object level 改 harness/skill/context(Darwin Gödel Machine、Meta-Harness 等);本文改 meta-level 探索控制器。
- 记忆复用:历史 → context/memory 是主流;本文 → simulator,闲置经验变成 meta 优化的可复用反馈。

## 对本工作区的启示

- **方法论可直接迁移到 kernel/系统优化任务**:discovery tree + replay simulator 的思路,正是 `~/source_code/original_performance_takehome/`(VLIW kernel 优化,已优化到 980 cycles)这类"agent 驱动 kernel 搜索"任务缺少的一层 —— 每次优化 session 的尝试历史都沉淀为树,新探索策略可先离线重放评估再上线,省掉重复跑昂贵评测。KernelBench 四项任务(尤其 ConvDiv/ConvMax 的 2.09×)说明该收益在 kernel 领域真实存在。
- **对 agent harness 设计**(`~/source_code/kimi-code/`):论文的防作弊约束清单值得借鉴 —— prefix-only 观测、禁止硬编码 cell id/绝对分数目标、`best_so_far`/`budget_spent` 只作记账不可入决策、replay trace 只能 between-round 读不能进 `solve()`。凡是让 LLM 写"策略/调度代码"的系统,都应显式声明这类信息边界。
- **对推理服务/infra 研究的间接关联较弱但存在**:其本质是把"日志"变成"环境",与 `~/source_code/vllm/`、`sglang/` 里的 trace-driven 调优、simulator-based 调度评估思路同源;若做 PD 分离/KV 调度的策略搜索,也可考虑用历史 trace 构建重放模拟器来低成本评估调度策略。
- **局限性**:simulator 只覆盖已观测搜索空间(replay 超出 trace 支撑不计奖励);实验全用 Gemini 系列,无跨厂商模型对照;beta 系数等超参固定。
