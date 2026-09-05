# DeepSeek-V3.2 技术报告总结(arXiv 2512.02556)

> 来源:`~/.cache/nanochat/knowledge/2512.02556/`(arXiv TeX 源码,入口 `main.tex`,正文在 `sections/`)。
> 论文标题:*DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models*(DeepSeek-AI)。
> 一句话:在 V3.1-Terminus 基座上引入 **DeepSeek Sparse Attention (DSA)**,并大幅扩张 RL 后训练算力 + 大规模 agentic 任务合成流水线;高算力变体 **DeepSeek-V3.2-Speciale** 在 IMO 2025 / IOI 2025 / CMO 2025 / ICPC WF 2025 均达金牌线。

## 1. 模型架构(§2 DeepSeek-V3.2 Architecture)

- 架构与 DeepSeek-V3.2-Exp 完全相同;相对 V3.1-Terminus **唯一改动是通过继续训练引入 DSA**。
- **DSA 两组件**(§2.1):
  - **Lightning indexer(闪电索引器)**:对 query token h_t 与历史 token h_s 计算索引分
    `I_{t,s} = Σ_j w^I_{t,j} · ReLU(q^I_{t,j} · k^I_s)`。索引器头数少、可用 FP8 实现,选 ReLU 是为吞吐。
  - **细粒度 token 选择**:每个 query 只取 top-k 索引分对应的 KV 条目做注意力,主注意力复杂度从 O(L²) 降到 O(Lk)。
- **DSA 在 MLA 下的实例化**:为了从 V3.1-Terminus 继续训练,DSA 基于 MLA 实现;kernel 层面要求每个 KV 条目被多个 query 共享(引 NSA 论文 yuan-etal-2025),因此采用 **MLA 的 MQA 模式**(latent 向量被该 query 的所有头共享)。附录 A 说明 MLA 的 MHA/MQA 两种模式:V3.1-Terminus 训练/prefill 用 MHA 模式、decode 用 MQA 模式。
- **推理成本**(§2.3):indexer 仍是 O(L²) 但比 MLA 便宜得多;短序列 prefill 用 masked MHA 模式模拟 DSA;成本曲线基于 H800(2 USD/GPU·h)实测。
- 官方参考实现随模型开源(HF deepseek-ai/DeepSeek-V3.2-Exp 下 inference 目录)。

## 2. 预训练 / 继续预训练(§2.1.1 Continued Pre-Training)——数据细节

**重要:本报告不是从零训练,V3.2 从 V3.1-Terminus 的 base checkpoint(上下文已扩到 128K)出发做继续预训练。** 原始基座(V3 系)的预训练 token 总量本报告未重复披露(见 V3 报告)。

继续预训练共两个阶段,**两个阶段的数据分布都与 V3.1-Terminus 的 128K 长上下文扩展数据完全对齐**(原文:"the distribution of training data is totally aligned with the 128K long context extension data used for DeepSeek-V3.1-Terminus")。

| 阶段 | 步数 | 每步 batch | 总 token | 学习率 | 训练内容 |
|---|---|---|---|---|---|
| Dense Warm-up | 1,000 | 16 条 × 128K | **2.1B** | 1e-3 | 冻结主模型,只训 indexer;目标是把主注意力(跨头求和、L1 归一化)分布 p 与 Softmax(I) 的 KL 散度最小化 |
| Sparse Training | 15,000 | 480 条 × 128K | **943.7B** | 7.3e-6 | 引入 top-k=2048 稀疏选择,全参数训练;indexer 输入从计算图 detach,indexer 只吃 L^I(只在被选集合 S_t 上对齐),主模型只吃 LM loss |

- **继续预训练合计 ≈ 945.8B tokens**(2.1B + 943.7B)。
- **Epoch/数据重复:论文未披露**(没有说明语料原始规模、训练几遍、高质量子集是否多 epoch)。
- **Mid-training/annealing:未以该名义披露**;上述继续预训练(对齐 V3.1-Terminus 的 128K 扩展数据)就是本报告唯一的"中途训练"阶段,数据构成未进一步细分。
- 性能对齐验证(§2.2):V3.2-Exp 与 V3.1-Terminus 在短/长上下文基准、ChatbotArena Elo(2025-11-10)上持平;AA-LCR、Fiction.liveBench 等第三方长上下文评测甚至更高。

## 3. 后训练(§3 Post-Training)

- 后训练同样用 DSA 稀疏注意力;流水线与 V3.2-Exp 相同:**specialist distillation + mixed RL**。
  - **Specialist 蒸馏**:写作/通用 QA + 六大专门域(数学、编程、通用逻辑推理、通用 agent、agentic coding、agentic search),thinking/non-thinking 双模式;每个 specialist 从同一 base 出发经大规模 RL 训练,再产出蒸馏数据;蒸馏模型与 specialist 的差距由后续 RL 抹平。
  - **Mixed RL**:GRPO;推理、agent、人类对齐合并为一个 RL 阶段(避免多阶段灾难性遗忘);推理/agent 用规则化 outcome reward + 长度惩罚 + 语言一致性奖励,通用任务用带逐 prompt rubrics 的生成式奖励模型。
  - **RL 算力 > 预训练成本的 10%**(§4.1 重申:"already exceeds 10% of the pre-training cost")。Speciale 变体只训推理数据、放松长度惩罚,并引入 DeepSeekMath-V2 的数据与奖励方法。
- **Scaling GRPO 的稳定化技术**(§3.1),全部针对 RL 规模化:
  1. **Unbiased KL Estimate**:修正 K3 估计器,用当前/旧策略的重要性采样比使 KL 梯度无偏(K3 在 π_θ ≪ π_ref 时给出无界大梯度,导致训练不稳);不同域用不同 KL 强度,数学域可弱惩罚甚至不加。
  2. **Off-Policy Sequence Masking**:rollout 大批次拆分多步更新天然 off-policy,叠加推理/训练框架实现差异;对"负 advantage 且新旧策略 KL 超阈值 δ"的序列加二值 mask M 置零,只 mask 负样本。
  3. **Keep Routing**(MoE):保存采样时推理框架的 expert 路由路径并在训练时强制复现,防止同一输入路由不一致引起活跃参数子空间突变;自 DeepSeek-V3-0324 起就在用。
  4. **Keep Sampling Mask**:top-p/top-k 截断采样会破坏重要性采样的动作空间一致性;保存采样时的截断 mask 并在训练时施加到 π_θ,保证新旧策略动作子空间相同。
- **Thinking in Tool-Use**(§3.2):
  - **思考上下文管理**:只有新 user 消息到来才丢弃历史推理内容;仅追加 tool 消息时推理内容全程保留;推理被移除时 tool call 及其结果历史仍保留。注意 Roo Code / Terminus 这类把工具交互模拟成 user 消息的框架吃不到该收益(Terminal Bench 2.0 的 46.4 分因此用 Claude Code 框架测得)。
  - **Cold-start**:靠系统提示词把推理与 tool call 缝合进同一轨迹(模板见附录 B 表 3–5)。
  - **大规模 agentic 任务合成**(§3.2.3):任务量表(Table 1)——code agent 24,667(真实环境/抽取 prompt)、search agent 50,275(真实环境/合成 prompt)、general agent 4,417(全合成)、code interpreter 5,908(真实环境/抽取 prompt)。
    - Search agent:多 agent 流水线从大规模 web 语料采长尾实体 → 建题 agent 按可调深度/广度用搜索工具探查 → 异构配置的多答题 agent 产候选 → 带搜索能力的验证 agent 多遍验证,只留"GT 正确且候选均可证伪"的样本;另混有用性 RL 数据,用 rubrics + 生成式 RM 打分。
    - Code agent:从 GitHub 挖数百万 issue-PR 对,启发式 + LLM 过滤(需 issue 描述、gold patch、test patch),自动环境搭建 agent 装依赖跑测试(JUnit 格式),只留 F2P>0 且 P2F=0 的环境;建成数万个可复现环境,覆盖 Python/Java/JS/TS/C/C++/Go/PHP。
    - General agent:自动环境合成 agent 产出 ⟨环境, 工具, 任务, 验证器⟩ 四元组,任务"难解易验"(如旅行规划,约束组合空间大但验证便宜);用 V3.2 做 RL 后只留 pass@100 非零的实例 → **1,827 个环境 / 4,417 个任务**。
    - 消融(§4.3 Synthesis Agentic Tasks):合成任务对 V3.2-Exp 只有 12% 通过率、对前沿闭源模型最高 62%,足够难;仅在合成 general agent 数据上做 RL(non-thinking)即可显著提升 τ²-bench / MCP-Mark / MCP-Universe,而只在 code+search 上 RL 无此收益 → 合成数据有跨域泛化。
- **Search agent 的测试时上下文管理**(§4.4):token 用量超 128K 窗口 80% 时触发 Summary / Discard-75% / Discard-all / Parallel-fewest-step;Discard-all 简单但效率与可扩展性俱佳(BrowseComp 67.6);Summary 平均扩到 364 步、60.2 分但效率低。

## 4. 评测结果(§4)

- 评测集:MMLU-Pro、GPQA-D、HLE(text-only)、LiveCodeBench、Codeforces、Aider-Polyglot、AIME 2025、HMMT Feb/Nov 2025、IMOAnswerBench、Terminal Bench 2.0、SWE-Verified、SWE Multilingual、BrowseComp(Zh)、τ²-bench、MCP-Universe、MCP-Mark、Tool-Decathlon;temperature 1.0、128K 窗口。
- V3.2 推理类与 GPT-5-high 相当、略逊 Gemini-3.0-Pro;与 K2-Thinking 分数相当但输出 token 显著更少。SWE-Verified 内部框架主分,跨 Claude Code/RooCode/non-thinking 复测 72–74。Search agent 因 128K 上限约 20%+ 用例超长,靠上下文管理拿分(无管理 51.4)。
- **V3.2-Speciale**(§4.2):放松长度约束换性能,多项基准超 Gemini-3.0-Pro(AIME 96.0、HMMT Feb 99.2、HMMT Nov 94.4、IMOAnswerBench 84.5、Codeforces 2701);**IOI 2025 金牌(492/600,第 10 名)、ICPC WF 2025 金牌(10/12,第 2)、IMO 2025 金牌(35/42)、CMO 2025 金牌(102/126)**;但 token 效率显著差于 Gemini-3.0-Pro。评测协议(附录 D):IOI 每题采样 500 候选→过滤→取最长 thinking 的 50 个提交;ICPC 每题 32 候选;IMO/CMO 用 generate-verify-refine 循环(同 DeepSeekMath-V2)。
- 局限(§5):总训练 FLOPs 少导致世界知识广度落后;token 效率(智能密度)仍需优化;复杂任务仍不及前沿闭源。

## 5. Infra / 系统工程要点

- **DSA kernel**:FP8 lightning indexer + top-k 稀疏选择;短序列 prefill 用 masked MHA 模式模拟 DSA;部署实测在 H800 集群(2 USD/GPU·h)。
- **RL infra**:推理框架直接返回采样概率 π_old 以同时度量"多步更新 off-policy"与"训练/推理框架实现差异"两类 off-policyness;Keep Routing / Keep Sampling Mask 都依赖推理侧把采样时的路由路径和截断 mask 传给训练侧 —— 推理与训练框架需要深度协同。
- **Agentic 环境 infra**:可执行 SWE 环境的自动搭建(依赖解析、JUnit 标准化输出、F2P/P2F 校验)、合成环境沙箱(bash + search 工具)、竞赛评测的多候选过滤流水线。

## 6. 与 ~/source_code/ 源码的联系

- **vLLM**:DSA 的注意力后端在 `vllm/v1/attention/backends/mla/` 下,如 `flashmla_sparse.py`(FlashMLA sparse decode)、`flashinfer_mla_sparse_sm90.py`、`rocm_aiter_mla_sparse.py`(ROCm)、`xpu_mla_sparse.py`;模型定义在 `vllm/model_executor/models/deepseek_v2.py`(V3.2 复用,经 config 区分 sparse)。论文的"top-k 稀疏选择 + FP8 indexer"正对应这些后端的 `topk_indices` 数据通路。
- **SGLang**:DSA 支持在 `sglang/python/sglang/srt/layers/attention/dsa/`(kernel/索引逻辑)与 `dsa_backend.py`,模型在 `srt/models/deepseek_v2.py`;`srt/mem_cache/sparsity/algorithms/deepseek_dsa.py` 还把 DSA 思想用于 KV cache 稀疏化;`test/registered/rl/test_return_indexer_topk.py` 对应"推理框架返回 indexer top-k / 采样概率"的训推协同需求。NSA(论文引用的 kernel 共享原则)在 `srt/layers/attention/nsa/`。
- **ktransformers**:`ktransformers` 对 DeepSeek-V3 系的 MLA + MoE CPU/GPU 异构卸载,是论文 MLA MQA 模式 decode 的低资源实现参照。
- **DeepGEMM**:DSA indexer 及 MoE 的 FP8 GEMM 依赖与 `source_code/DeepGEMM/`(Hopper FP8 GEMM)同源,论文中 indexer 可 FP8 实现即建立在这类 FP8 kernel 之上。
- **mooncake / nixl**:论文 §4.4 的上下文管理(Summary/Discard-all)与 PD 分离场景下 KV cache 传输(mooncake 的 KV cache 服务、nixl 的传输抽象)互补——长上下文 agentic rollout 的 KV 复用/卸载可落到这两个仓库的机制上。
