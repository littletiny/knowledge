# GLM-5: from Vibe Coding to Agentic Engineering(arXiv 2602.15763)论文总结

- 机构:Zhipu AI & Tsinghua University;代码/模型:https://github.com/zai-org/GLM-5
- 定位:GLM-4.5/4.7 之后的旗舰开源权重 MoE 模型,主打 ARC(Agentic / Reasoning / Coding),核心卖点是 DSA 稀疏注意力降本 + 全异步 agentic RL 后训练体系。对标 Claude Opus 4.5 / GPT-5.2 / Gemini 3 Pro。
- 注意:本报告主体是 **GLM-5**(及 GLM-4.7 对比),**未披露 GLM-5.2/5.3 的预训练细节**。GLM-5.2 信息来自 z.ai 官方博客(见文末);GLM-5.3 与本路线同 base,仅扩大 post-training(博客/论文均未给 5.3 预训练数字)。

## 1. 模型架构(Section 2.1 + Appendix A)

- 总参 744B / 激活 40B(GLM-4.5 为 355B/32B);80 层(Appendix 表:3 dense + 75 MoE + 1 MTP;本地 config `~/model-knowledge/configs/glm-5.2` 显示 `num_hidden_layers=78`,即 3+75)。
- MoE:256 routed experts(比 GLM-4.5 的 160 增多)+ 1 shared,每 token 激活 8 个,MoE intermediate 2048。层数从 GLM-4.5 减少以降低 EP 通信开销。
- Attention:MLA(Multi-head Latent Attention,DeepSeek-V2 式)+ DSA 稀疏化。
  - 关键改动 1 **Muon Split**:直接对 MLA 的 576 维 latent KV 用 Muon 训练追不上 GQA-8;把 $W^{UQ},W^{UK},W^{UV}$ 按 head 拆成小矩阵分别做正交化,使 MLA 追平 GQA-8,且 attention logits 训练期无需 clipping 也稳定(Table 1)。
  - 关键改动 2 **MLA-256**:训练/prefill 时 MLA 是 MHA 形态,把 head dim 从 192 提到 256、头数减 1/3(64 heads),保持训练计算量和参数量不变但降低 decoding 计算(decoding 时 MLA 是 576 维点积,高于 GQA 的 128 维)。Appendix 架构表:QK head dim 192(=128 nope + 64 rope)、V head dim 256、Q LoRA 2048、KV LoRA 512、indexer 32 heads × 128 dim。
- **MTP 参数共享**:训练时 3 个 MTP layer 共享参数,推理时 4 步投机解码,accept length 2.76,高于 DeepSeek-V3.2 的 2.55(Table 2)。
- Vocab 154880。INT4 QAT 在 SFT 阶段做,配套量化 kernel 保证训练与离线权重量化 bitwise 一致。

### DSA 继续预训练(Section 2.1.1)
- 沿用 DeepSeek-V3.2 的 "dense warm-up + sparse adaptation" 两阶段,从 mid-training 结束的 MLA base 上继续做:
  - Warm-up:1000 steps × 每步 14 条 × 202,752 tokens(≈ 2.84B tokens),最大 LR 5e-3(Appendix:5e-3 降到 2e-4);
  - Sparse adaptation:沿用 mid-training 的数据和超参,**20B tokens**,恒定 LR 1e-5。
- 对比 DeepSeek-V3.2 的 943.7B tokens,GLM-5 只用 ~23B 就适配到与 MLA 模型持平(Table 3 长上下文基准 + 相同 SFT 数据下 loss/评测打平,Figure 3)。
- Section 2.1.2 消融(GLM-9B,190B tokens 继续训练,1:1 高效层:全注意力层):搜索式 SWA 层选择(beam search,RULER@16K 上搜,得到的 pattern 泛化到各长度)> GDN/SimpleGDN > 固定交替 SWA(灾难性掉点);但所有方案在长上下文细粒度检索上仍有 5-7 分固有差距,而 DSA 的 lightning indexer 是 token 级稀疏、"无损",可用于全部层。GLM-4.7-Flash 小规模验证:warmup-only 已保住大部分性能,150B tokens 联合训练后基本追平基线(Table 5)。

## 2. 预训练数据与流程(Section 2.2、2.3)

**总量:base model 全阶段共 28.5T tokens**(Section 2 开头;Introduction 说 pre-train 语料 27T + mid-training 三阶段 1T+500B+50B = 1.55T,合计 ≈28.5T,数字自洽)。

- **Pre-training(27T)**:在 GLM-4.5 pipeline 基础上:
  - Web:新增基于句向量 embedding 的 DCLM 分类器挖掘高质量数据;World Knowledge 分类器(Wikipedia + LLM 标注优化)从中低质数据蒸馏长尾知识。
  - Code:刷新代码托管平台快照 + 更多含代码网页,**fuzzily dedup 后 unique tokens 增加 28%**;修 Software Heritage 元数据对齐问题;低资源语言(Scala/Swift/Lua 等)专用分类器;沿用 GLM-4.5 quality-aware 采样。
  - Math & Science:网页/书/论文,LLM 打分只留高教育价值内容,长文档用 chunk-and-aggregate 打分;严格过滤合成/AI 生成/模板数据。
- **Mid-training(1.55T,Section 2.3)**:三阶段渐进扩展上下文:**32K(1T tokens)→ 128K(500B)→ 200K(50B)**。后阶段上采样长文档与合成 agent 轨迹。
  - SWE 数据:repo 文件 + commit diff + issue + PR 拼接成统一序列;放宽 repo 级过滤得到 ~10M issue-PR 对、加强 issue 级质量过滤,过滤后 issue-PR 部分 **~160B unique tokens**。
  - 长上下文数据:自然数据(书/论文/通用语料,PPL/去重/长度多级过滤,知识密集域上采样)+ 合成数据(NextLong/EntropyLong 式长程依赖构造、相似文本 interleaved packing 缓解 lost-in-the-middle);200K 阶段加少量 MRCR-like 数据。经验:200K 阶段也反过来提升 128K 内性能。
- 超参(Appendix A):Muon + cosine decay + batch warmup;pre-train LR warmup 到 2e-4、decay 到 4e-5;mid-train LR 4e-5 → 1e-5 线性。
- **未披露**:epoch/pass 数、原始(未去重)语料总规模、高质量子集的多 epoch 策略(只有 "up-sample" 表述,无具体倍数)、数据配比百分比。

## 3. 后训练(Section 3、4)

流程:SFT → Reasoning RL → Agentic RL → General RL → On-Policy Cross-Stage Distillation(全程防遗忘)。

- **SFT**:三类数据(General Chat / Reasoning / Coding & Agent),agent 与 coding 数据大幅扩量;上下文扩到 202,752 tokens;三种思考模式:Interleaved Thinking(每次回复/工具调用前思考)、Preserved Thinking(跨轮保留 thinking block)、Turn-level Thinking(逐轮开关)。轨迹中错误段保留但 loss mask,学纠错行为。
- **Reasoning RL**:GRPO + IcePop 处理训练-推理分布不一致(去掉 KL 项;β=2, ε_low=0.2, ε_high=0.28;group 32 / batch 32,完全 on-policy)。数学/科学/代码/TIR 四域混合,按 GLM-4.7 难度过滤,按域配 judge。
  - **DSA RL 要点**:indexer top-k(k=2048)太大无法像 MoE routing replay 那样存索引;发现必须用**确定性的 `torch.topk`**(SGLang 里 CUDA/TileLang 的非确定性 top-k 会让 RL 几步后崩溃、熵骤降),且 RL 期间默认冻结 indexer 参数。
- **Agentic RL(Section 4)**:全异步解耦架构——训练/推理引擎分离,推理持续产轨迹,每 K 次梯度更新回推权重;>1k 并发 rollout 的 Multi-Task Rollout Orchestrator(微服务注册、统一 message-list 表示)。稳定性三件套:
  - **TITO(Token-in-Token-out)** gateway:不重 tokenize,避免文本往返的边界错位;
  - **Direct Double-sided IS**:直接复用 rollout logprob 作 behavior policy,`r_t = π_θ/π_rollout`,区间 [1-ε_l, 1+ε_h] 外的 token 整段 mask;
  - 丢样本:版本过旧(w′−w₀>τ)的轨迹丢弃;环境崩溃样本剔除,GRPO 组内不足半数有效则整组丢弃。
  - **DP-aware routing**:rollout ID 一致性哈希到固定 DP rank,多轮共享前缀 KV cache,配轻量再均衡。
  - 环境规模:10K+ 可验证 SWE 环境(RepoLaunch pipeline,9 种语言)、终端任务(Harbor 格式,种子合成 + web 语料合成,Docker 构建成功率 >90%)、多跳搜索(WKG 知识图谱 + 三阶段难度/正确性过滤)。
  - Search agent 推理期上下文管理:keep-recent-k(k=5,BrowseComp 55.3→62.0)+ Discard-all 混合(T=32k),BrowseComp w/ CM 达 75.9。
- **General RL**:三维目标(基础正确性 / 情感智能 / 任务质量)+ 混合奖励(rule / ORM / GRM)+ 人类撰写回答作为风格锚点。
- **On-Policy Cross-Stage Distillation**:最后阶段,以各前序阶段 checkpoint 为 teacher,advantage 换成师生 logprob 差;GRPO group=1、batch=1024。
- 另有 Slide 生成 RL(三级奖励:静态 HTML 属性 / 运行时渲染属性 / 视觉感知;遇到 reward hacking 如硬截断,通过渲染器实现修复;16:9 合规率 40%→92%)。

## 4. 系统工程(Section 2.4、3.6、5)

预训练 infra:
- 灵活 MTP 放置(embedding/transformer 放倒数第二 stage,output 与主 output 合置末 stage 做参数共享,均衡显存);
- Pipeline ZeRO2 梯度分片(每 stage 只存 1/dp 梯度 + 双缓冲滚动累加);
- Muon 分布式优化器零冗余通信(all-gather 仅限本分片,计算与通信 overlap);
- pipeline 激活 offload 到 host、按层粒度、与计算 overlap,配合细粒度重算;
- 输出投影/CE loss 按序列 chunk 计算降峰值显存;
- 延迟 weight grad 计算降气泡;长序列负载均衡(workload-aware 重排 + 动态 attention 重分配 + 可变大小 CP 组 + 分层 all-to-all)。

RL infra(slime 框架,Section 3.6):
- 优化目标是 **tail latency** 而非吞吐:FP8 rollout、MTP(小 batch 长尾收益大)、PD 分离(DP-attention 下 prefill 不干扰 decode);
- 多节点推理(EP64 + DP64,8 节点)提供足够分布式 KV cache,DP-attention 避免 KV 跨 rank 拷贝;
- server-based rollout(HTTP API 解耦)、心跳容错 + 路由层摘除故障服务器。

国产芯片适配(Section 5,7 平台:华为昇腾、摩尔线程、海光、寒武纪、昆仑芯、沐曦、燧原;以 Atlas 800T A3 为例):
- W4A8 混合精度(Attention/MLP W8A8,MoE experts W4A8),QuaRot + Flex_AWQ_SSZ,单节点装下 750B;
- 定制融合 kernel:Lightning Indexer(打分+ReLU+TopK 单 kernel)、Sparse Flash Attention、MLAPO(13 个预处理小算子合一);
- 适配 vLLM-Ascend 与 SGLang:异步调度(overlap D2H 采样拷贝)、RadixCache/Prefix Cache、Attention DP + MoE EP + FlashComm、MTP。单国产节点性能可比肩双 GPU 国际集群,长序列部署成本降 50%。

## 5. 评测要点(Section 5 + Appendix B)

- Base:GLM-5-Base 在 SimpleQA/EvalPlus/LiveCodeBench-Base 等超 GLM-4.5/DeepSeek-V3/Kimi-K2 Base,但 GSM8K/MATH 反而低于 K2(68.8/56.4,论文未解释)。
- ARC:HLE 30.5 / HLE(w/ tools) 50.4、SWE-bench Verified 77.8、Terminal-Bench 2.0 56.2(verified 60.7)、BrowseComp 62.0(w/ CM 75.9)、τ²-Bench 89.7、Vending-Bench 2 $4,432;开源权重 SOTA,Artificial Analysis Intelligence Index 50 分(首个到 50 的开源模型),LMArena Text/Code 双榜开源第一。
- 自建 CC-Bench-V2(前端 Agent-as-a-Judge / 后端单元测试 / 长程 repo 探索与链式任务):整体接近 Claude Opus 4.5,前端 ISR 与链式任务仍有差距。
- Easter egg:GLM-5 曾以 "Pony Alpha" 匿名上 OpenRouter,25% 用户猜是 Claude Sonnet 5。

## 6. 与 ~/source_code/ 各仓库的联系

- **sglang/**:DSA 的 lightning indexer 实现在 `python/sglang/srt/layers/attention/dsa/dsa_indexer.py`(`class Indexer`,top-k 索引选择,`nsa/nsa_indexer.py` 已 deprecate 转 shim);论文 Section 3.2 明确提到 "SGLang's DSA Indexer 的 CUDA 非确定性 top-k 在 RL 中导致崩溃,训练侧改用 `torch.topk`"——这是读 sglang DSA 代码时的一个关键伏笔。SGLang 也是被 GLM-5 官方适配昇腾的两个推理引擎之一(Section 5)。
- **DeepGEMM/**:DSA indexer 的核心打分 kernel 即 `fp8_paged_mqa_logits` / `fp8_mqa_logits`(`csrc/apis/attention.hpp`、`deep_gemm/include/deep_gemm/impls/sm90_fp8_mqa_logits.cuh` 等)——把 MLA 的 MQA 模式 KV 打分做成 FP8 paged kernel,正是论文中 MLA "decoding 时用 MQA 模式" 与 indexer 高吞吐的工程基础。
- **vllm/** & **vllm-ascend/**:GLM-5 的国产芯片推理适配直接基于 vLLM-Ascend(论文 Section 5);vLLM 侧已有 `transformers_utils/configs/glm5_next.py`(glm5_next 架构,即 GLM-5.3-Flash 系)与 deepseek_v32(DSA 同源)相关支持。DP-aware routing / RadixCache / PD 分离均为 vLLM/SGLang 现有机制。
- **mooncake/**:论文 slime 的 PD 分离 + KV cache 亲和路由与 Mooncake 的 KV cache 传输/PD 解耦(TransferEngine、mooncake-store)是同一条技术路线,可对照读。
- **ktransformers/**:GLM-5 这类 744B MoE + INT4 QAT(训练推理 bitwise 一致)的模型正是 ktransformers CPU/GPU 异构 MoE 推理的目标场景。
- slime(RL 框架)未在本地 clone,如需深入可 clone THUDM/slime 到 ~/source_code/。

## 7. GLM-5.2 / GLM-5.3 补充(非论文内容)

来自 [z.ai GLM-5.2 博客](https://z.ai/blog/glm-5.2):
- GLM-5.2 相对 GLM-5.1:**1M token context**(config 确认 `max_position_embeddings=1048576`);新架构改动 **IndexShare**(每 4 个稀疏注意力层共享同一 indexer,1M 下每 token FLOPs 降 2.9×);MTP 改进使投机解码接受长度提升至多 20%;MIT 协议开源。
- "substantially expanded 1M-context training for coding-agent scenarios"——**未披露具体 token 数、数据构成或 epoch 数**。
- 本地 config 对比(`~/model-knowledge/configs/glm-5.2` 与 `glm-5.3`):两者 `glm_moe_dsa` 架构**完全相同**(78 层、hidden 6144、256 experts、indexer 32×128/topk 2048、1M context),佐证 5.2/5.3 与 GLM-5 同 base 路线,差异在 post-training。**GLM-5.3 的预训练信息在论文与博客中均未披露。**
