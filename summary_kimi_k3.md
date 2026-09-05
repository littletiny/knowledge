# Kimi K3: Open Frontier Intelligence 技术报告总结

- arXiv: 2607.24653(Kimi Team 技术报告),TeX 源码缓存于 `~/.cache/nanochat/knowledge/2607.24653/`(入口 `main.tex`,章节拆分为 `1-introduction.tex` ~ `8-conclusion.tex` + `appendix.tex`)。
- 模型主页:https://huggingface.co/moonshotai/Kimi-K3;本地 config 见 `~/model-knowledge/configs/kimi-k3/config.json`。

**一句话**:Kimi K3 是 2.8T 总参数 / 104B 激活参数的原生多模态 MoE 模型,支持 1M token 上下文,架构核心是 KDA(Kimi Delta Attention)混合注意力 + AttnRes(注意力残差)+ Stable LatentMoE(896 路由专家 / 每 token 激活 16),整体 scaling 效率约为 K2 的 2.5 倍。评测上仅次于 Claude Fable 5 和 GPT-5.6 Sol,稳定领先其他开源/闭源模型(GLM-5.2、Claude Opus 4.8、GPT-5.5)。权重全量开源。

---

## 1. 模型架构(§2,`2-model-architecture.tex`)

设计主线是沿三个维度扩大信息流:序列长(混合注意力)、深度(AttnRes)、宽度(LatentMoE)。关键结构参数(K2 → K3 对比见 Table 1):层数 61 → 93(69 层 KDA + 24 层 Gated MLA,block 内 3:1 混合,末尾额外 1 层 MLA 保证最后一层是全局注意力);总参数 1.04T → 2.78T;激活参数 32.6B → 104.2B;hidden 7168 不变;路由专家 384 → 896;top-k 8 → 16;共享专家 1 → 2;注意力头 64 → 96;词表 160K;MTP 层 1 层;训练上下文 128K → 1M。

### 1.1 Hybrid Attention:KDA + Gated MLA(§2.1)
- **KDA**(Kimi Delta Attention,源自 Kimi Linear `team2025kimi`):delta-rule 线性注意力,带逐通道遗忘门。递归式 `S_t = (I − β_t k_t k_t^T) Diag(α_t) S_{t-1} + β_t k_t v_t^T`;chunk 内并行、chunk 间递归(chunkwise 形式沿用 Kimi Linear 的 UT transform)。q/k/v 投影用 ShortConv + Swish,q/k 加 L2Norm,衰减由低秩投影 + 每头 bias 产生逐 key 通道的 logit。
- **K3 的两处改动**:
  1. **下界衰减(lower-bounded decay)**:Kimi Linear 用无界的负 Softplus 映射 `g = −e^A Softplus(z)`,K3 改为带缩放 sigmoid `g = g_min·Sigmoid(e^{A_h} z)`,固定 `g_min = −5`,使保留因子 α > e^{−5} ≈ 6.7e-3。这样 16-token tile 内累积 log 衰减在 (−80, 0),倒放缩因子 < e^80,在 BF16 动态范围内 → 对角 tile 不再需要逐位置对计算,全部可用 dense Tensor Core matmul,消除了 Kimi Linear 的主要 chunk 内瓶颈。
  2. **全秩输出门**:输出门从 Kimi Linear 的低秩参数化改为输入依赖的全秩投影 `y = W_o[Sigmoid(W_g x) ⊙ RMSNorm(õ)]`。
- **Gated MLA**(§2.1.2):周期性全局注意力层用 MLA(DeepSeek-V2 式低秩 KV 压缩),K3 的改动:(a) 所有 MLA 层用 **NoPE**(不用 RoPE/位置编码),位置信息完全交给 KDA 层的递归门控和衰减 —— 这样扩上下文时无需重调 RoPE base 或 YaRN;(b) 加输入依赖的全秩 channel-wise 输出门(同 KDA 的参数化);(c) 为修正 flash attention 的有偏舍入误差,训练时注意力输出保持 FP32,并重新设计 kernel 把输出 tile 与 KV staging buffer 复用以腾出 shared memory。

### 1.2 Attention Residuals(AttnRes,§2.2)
把"深度"当作"序列"来做注意力:每层用一个可学习 pseudo-query `q_l = w_l`,对 embedding 和所有前驱层输出做 softmax 注意力权重(`φ(q,k)=exp(q^T RMSNorm(k))`),加权求和得到该层输入。完整版复杂度 O(L²d) 可接受,瓶颈是 O(Ld) 的显存/跨 stage 通信。**Block AttnRes**:L 层分成 N 个 block,block 内输出求和压缩成单个表示,block 间做全注意力;K3 用 12 层/block 分 8 个 block(加 embedding 共 9 个源),显存/通信开销降到 O(Nd),且推理时可用 online softmax 合并块间并行结果与块内部分和。

### 1.3 Stable LatentMoE(§2.3)
- LatentMoE:共享专家走全宽 d=7168;路由路径先 `W↓` 降到 latent 宽 ℓ=3584(0.5×),再在 latent 空间跑 896 个路由专家(top-16,稀疏度 56),聚合后经 RMSNorm 再由 `W↑` 升回全宽。共享专家固定 2 个。本地 config 中 `latent_moe_use_norm: true`、`hidden_act: "situ"` 与此对应。
- 极端稀疏度放大两个失效模式,对应三个稳定化组件:
  1. **Normalized LatentMoE**:路由聚合后、`W↑` 前插 RMSNorm,抑制尺度漂移(还稳定提升 val loss 和下游指标)。
  2. **SiTU-GLU**(Sigmoid Tanh Unit GLU):替代 SwiGLU,对 gate 分支的线性因子和 up 分支分别做 softcap `β·tanh(x/β)`,β₁=4、β₂=25;输出严格有界 |f|≤β₁β₂=100,抑制激活 outlier/溢出,同时保留 SwiGLU 近原点的一阶形状。
  3. **Quantile Balancing(QB)**:替代 DeepSeek-V3 的固定步长 bias 更新(`b += γ·sign(...)`)。每步用 Top-(k+1) 的 cutoff α_i,把每个专家的新 bias 直接设为 margin 的 (1−k/n) 分位数,使每个专家恰好收到目标负载 q=mk/n 个 token;推理时冻结 bias。全局 batch 上百万 token 的精确分位数不可行 → 用**直方图估计**:各 rank 统计 bin 计数,一次 all-reduce(每专家仅几百 bin)即得全 batch 池化分位数。附录给了从最优平衡分配(BASE Layers 视角)出发的推导和误差界。

### 1.4 Native Vision:MoonViT-V2(§2.4)
- 原生多模态:文本/图像/视频在同一 backbone、同一 NTP 目标下从头联合训练,无事后对齐阶段。
- **MoonViT-V2 完全从零用 next-token prediction 训练**,不再用 SigLIP 等对比学习预训练初始化(K2.5 的做法)——原因是稳定性:SigLIP 初始化的 MoonViT-3D 梯度范数持续偏高且频繁尖峰,而 from-scratch 的 V2 全程稳定;且评测上打平 SigLIP 基线,说明规模上对比预训练初始化并非必要。
- 27 层、约 0.4B 参数 ViT(patch 14,12 头),RMSNorm、去全部 bias;图像/视频共享参数,注意力分解为帧内空间 + 帧间时间两遍,时间维做 pooling;pixel-shuffle 2×2 下采样把视觉 token 数减 4 倍,支持最高 3584×3584 输入。

### 1.5 Per-Head Muon(§2.5)
矩阵参数沿用 Muon 优化器;注意力 Q/K/V 投影改为**逐头** Newton–Schulz 正交化(沿 head 维切分动量矩阵分别正交化),避免大梯度头主导共享更新方向,各头更新尺度均衡;同时 NS 迭代在瘦高的 per-head 块上更便宜。

---

## 2. 预训练(§3,`3-pre-training.tex`)

### 2.1 数据(§3.1)
- 文本语料覆盖四个主域:**Web Text、Code、Mathematics、Knowledge**;视觉语料覆盖 captions、交错图文、OCR、感知、视频、视觉编程(SVG、3D 资产、Webpage、Game、CAD 等"代码 + 渲染结果"的程序化多模态数据大幅扩量)。管线基于 K2 并在 K2.5 上精炼。
- 每个域用规则启发式 + 分类器质量打分 + 去重过滤,域采样率由小模型消融确定;**知识与数学语料沿用 K2 的 rephrasing 配方**(风格/视角多样化 prompting、chunk 级自回归生成、对照原文做保真校验)——即对高质量数据做改写增广。
- 视觉数据沿用 K2.5 的分类法,坐标监督同时给绝对坐标和 [0,1] 归一化坐标。

### 2.2 Scaling Law(§3.2)
- 架构/数据/训练改动改变了最优训练区间,因此重做 scaling law 搜索,调 batch size、学习率、**TPP(tokens-per-parameter)** 和 model shape;OOD 留出验证集上拟合的曲线显示相对 K2 约 **2.5× scaling 效率**。
- 在各自独立调参的前提下,**cosine decay 稳定优于 WSD**(作者指出共享超参对比不公平,因两个 schedule 的最优 peak LR / batch size 差异很大)。

### 2.3 训练配方(§3.3)
- 原生多模态:语言与视觉 token 从训练第一步起就在同一 NTP 目标下交错联合优化。
- Per-Head Muon + K2 的 weight clipping;QB 做负载均衡;**cosine LR schedule,1% 线性 warmup,weight decay 0.1 全程**。
- 预训练上下文先从 **8K** 开始,后续阶段扩到 **64K**。

### 2.4 长上下文扩展(§3.4)
- NoPE 架构免位置编码修改,直接外推到 1M。
- 长文档/视频有专门清洗管线(精确+模糊去重、视频帧感知哈希、质量过滤、结构校验);**因真正长且连贯的文档/视频稀缺,在 cooldown 阶段对长上下文数据做上采样(upsample)**,并**合成长上下文数据**——把多模态文档和子任务精心排列拼接,使任务必须利用散布在整个 1M 上下文中的信息才能解,防止注意力退化为局部模式。
- 渐进式四阶段课程:预训练期 8K → 64K,**cooldown 期 256K → 1M**;把昂贵的长序列计算集中在总预算的小部分内。1M 训练的序列维切分依赖 KDA Context Parallelism(§5.1.2)。

> ⚠️ **关键空白:本报告完全没有披露预训练总 token 数、原始语料 token 规模、epoch/pass 数、数据重复利用策略的细节,也没有给出 cooldown/annealing 阶段的 token 数与数据配比**。TPP 只作为 scaling law 里被重调的超参被提及,未给数值。论文只定性描述了数据域构成、过滤/rephrasing 流程,以及 cooldown 期长上下文数据的上采样与合成。这与 K2 技术报告(明确给出 15.5T tokens)的披露粒度相比明显收紧。

---

## 3. 后训练(§4,`4-post-training.tex`)

三阶段范式:**SFT 冷启动 → 分域分 effort 的 RL 专家 → Multi-Teacher On-Policy Distillation(MOPD)合并为单一模型**。

- **SFT**(§4.1.1):用前代 Kimi 分域专家模型合成 agentic 轨迹,多阶段验证 + human-in-the-loop 标注;所有数据用 **XTML**(eXtensible Token Markup Language,XML 同构但结构边界全部是特殊 token `[open]/[sep]/[close]`,附录 A.1)序列化;**从 SFT 起就做 QAT**(MoE 专家权重 MXFP4、激活 MXFP8,非专家模块保持高精度,见 §4.1.4)。
- **RL**(§4.1.2):三个大域 × 三个 reasoning effort{low, high, max} = **9 个专家模型**。域:general tasks(通用体验/视觉/推理/faithfulness/搜索/知识工作)、general agents(长程助理、deep research、段落级写作)、coding agents(SWE、coding 体验、kernel、webdev)。算法上扩展 partial rollout:每次迭代对 N 个 prompt 各采样 K 条,完成比例 λ 即暂停生成,未完成轨迹入队下一迭代优先恢复(靠沙箱基础设施);per-token 正则化把策略更新约束在局部邻域,容忍由此产生的极端 off-policy(轨迹跨多个迭代、数据高度陈旧)。**Reasoning Effort RL**:按问题设 token 预算 b₀(x),超 τ·b₀(x) 的轨迹奖励覆盖为 −1;先训 max 版再退火 τ 得到 high/low。非可验证任务用 Agentic GRM(锦标赛式二元比较 + 强制 rubric 流程 + 基于预算的啰嗦度惩罚,超 σ·ℓ₀ 直接判负)。
- **MOPD**(§4.1.3):对采样到的 (域, effort) 组合用对应教师做逐 token on-policy 蒸馏,reward 为 clip 过的 log 概率比(sg 停梯度),天然接入 RL 框架和 partial rollout;实验发现更细粒度 top-k 蒸馏无收益。
- **部署感知后训练**:MXFP4 QAT 贯穿 SFT+RL 全程,rollout 与训练同一量化方案消除 train-inference mismatch;**Draft Model Fine-Tuning**:预训练的 MTP 层结构即 EAGLE-3 草稿层,冻结 target、只训 draft 层和特征融合投影(融合第 1/4/最后一个 AttnRes block 的输出),训练时 7 步 unroll,且直接优化 LK loss(= −log 接受率)而非 KL 代理。
- **RL 环境**(§4.2):统一白盒环境(可组合出 Kimi Code、Claude Code、Codex、OpenClaw、Hermes 等各种 harness,防过拟合单一 harness);知识图谱引导的任务合成(agent 递归扩 DAG 知识图谱 → 采样节点检索真实材料 → 合成任务);可验证 agentic 任务(多步搜索、投行/数据分析/法律等专业工作、Python-in-the-loop 视觉推理);kernel 优化任务(CUDA/Triton/CuTe/Gluon/ThunderKittens/TileLang,正确性 + 对专家实现/roofline 的性能分 + 反 reward-hacking 检测);个人助理任务(Gmail/Notion/Slack/Canvas 模拟应用,多日持久环境,单次 rollout 可达数千次工具调用、百万 token);Autonomous Execution Tasks(verify-in-the-loop,黑盒系统复刻/量化因子发现/税务审计,公开+隐藏 verifier 防 hacking);webdev 任务(确定性检查 + 模型评审)。

---

## 4. Infra / 系统工程(§5,`5-infrastructure.tex`)

### 4.1 KDA 的算法-系统协同设计(§5.1)
- **FlashKDA**:CUTLASS chunkwise kernel,把 chunk 内计算与跨 chunk 状态传播流水重叠(token 并行 stage + head 并行 recurrence 独立调度),显著快于 Triton 参考;训练与推理 prefill 共用,已作为 flash-linear-attention(FLA)的后端自动分发。
- **设备内 CP**:线性注意力的段状态转移可独立于入口状态计算、事后精确复合 → SM 级 CP planner 在单卡内切分序列并行算段转移再合并(基于 DeltaNet CP 的工作)。
- **KDA Context Parallelism(KCP,§5.1.2)**:vanilla 线性注意力 CP(LASP)靠从 S=0 出发的局部状态直接求和,但 KDA 的 delta rule 含依赖入口状态的转移矩阵 M_t,不能直接求和。KCP 把每段分解为两个本地可算量——累积转移 `M^{T←1}` 和从零出发的局部状态 `S̃`——二者可结合(associative),用一次定长 **all-gather** + prefix scan 重建每个 rank 的入口状态,通信量与序列长无关,计算线性扩展。实现已合入 FLA PR #691。

### 4.2 3T 级预训练基础设施(§5.2)
- 并行策略:PP(virtual stages)+ EP + ZeRO-1 DP + Pipeline ZeRO-2 梯度分片(来自 GLM-5)+ CP;共享专家在 EP rank 间复制,all-to-all 与计算重叠。
- **MoonEP**(开源:github.com/MoonshotAI/MoonEP):完美负载均衡的 EP 方案。每 rank 恰好收 S×K 个 token → 各 rank 计算量完全一致、计算形状静态已知(消除每层 host 同步)。核心定理:**每 rank 预留至多 E/R 个冗余专家槽即可保证平衡方案必然存在,且该界基本紧致**(附录给了构造性证明和紧性例子);冗余专家在线规划(GPU planning kernel,离线用 ILP 求精确解做参照)、前向预取、反向梯度经本地 reduce buffer 回落 home rank。零拷贝:fused permute/unpermute,规划 kernel 预计算每个 token 的目的地,通信 buffer 视图直接交给计算;最坏情况下 DeepEP 需要 S×K×R 的 buffer,MoonEP 只需固定 S×K。专家 GEMM 用 workload-aware 调度器(解析成本模型 + 离线 autotune 标定系数);共享专家 GEMM 放单独 stream 重叠。
- **显存高效训练**(§5.2.2):统一激活管理器(重算/量化/offload 作为张量级可插拔存储策略组合,全部激活走单一显存池);大部分激活用 block-wise FP8 量化 + offload/远程 offload;MoE 梯度改写(借鉴 SonicMoE)消掉对前向 output 的依赖,group GEMM 前向只存 dispatch 输入、反向重算 dispatch 并与通信重叠;Block AttnRes 的块表示在边界层只生成一次、跨层共享,AttnRes 计算整体包 checkpoint,PP 间只增量传输新生成的块(cache-based pipeline communication,理论显存下界);PP rank 间激活不均衡 → 用 **Mooncake Transfer Engine** 把激活远程 offload 到其他 PP rank 的显存;Pipeline ZeRO-2 梯度分片后存 CPU(双 grad buffer 留在 GPU);**Muon 正交化改 P2P**:分布式优化器按 DP rank 分片参数,每 rank 只用 P2P 拉取自己拥有参数的完整分片做 NS 正交化(而非全参数 all-gather),按 model-chunk 粒度流水隐藏通信。
- **多模态编码器优化**(§5.2.3):大图/长视频沿 patch 维切到多设备,gather-KV 算注意力,CP 组再分子组做负载均衡;ViT 计算分解后塞进 interleaved 1F1B 的 PP 气泡(K2.5 的 DEP 的进一步演进),视觉编码器开销基本隐藏。

### 4.3 1M Agentic RL 基础设施(§5.3)
- Co-located RL(同集群训推一体,单次 1M RL 实验只用几百张 GPU)+ partial rollout;**外部 KV cache 池**:活跃 decode 块留 GPU,闲置可复用前缀写回 CPU DRAM(write-back,复用前预取),KDA 状态与 MLA KV 块同生命周期一起 offload/预取;训练态(权重+优化器)迭代结束后 offload 到 NVMe 给 DRAM 腾地方。
- Rollout auto-throttling:按活跃/排队请求数、KV 利用率动态控并发。
- 参考模型等 forward-only 非策略模型放 CPU,前向时把权重流式物化到策略模型的 FP32 梯度 buffer(两个 VPP chunk 槽位,一算一预取)。
- **AgentENV 沙箱**(开源:github.com/kvcache-ai/AgentENV):Firecracker microVM 隔离(agent 可挂盘/跑容器/起 VM);增量 checkpoint/resume(只存脏页,checkpoint 133ms / resume 49ms);Pause/Resume(等待推理期间零资源,占沙箱寿命可达 98%)、Fork(无副作用 reward judging)、Snapshot;OverlayBD 镜像 + ublk + 存储层共享 + P2P 传输,秒级启动数万个沙箱,内存超配 6.5×。全程训练评测共创建 **51,219,741 个沙箱、1,505,678 个镜像**。

### 4.4 推理与在线服务(§5.4)
- **KDA 感知前缀缓存**:KDA 定长递归态与 MLA 分页 KV 打包进同一分页池(统一页字节大小,共享分配/引用计数/驱逐);PD 分离下 TP 度不同时传输路径上做重排。前缀哈希用细粒度 hash block(512 token)而物理块保留粗粒度(1024–6144 token);KDA checkpoint 只存在 MLA hash 端点的稀疏子集(对话轮次边界保留);两段式查找(MLA 整物理块链式哈希 → 块内 hash 端点回退;KDA 要求所有 cache group 在候选边界都有 checkpoint),使命中最长边界是 512 的倍数而非物理块倍数——混合架构前缀缓存达到与全注意力相同的通用性。
- **Kernel**:KDA decoding 下 MTP 投机解码的状态回滚问题 → 只缓存 draft token 的投影输入(远小于状态),被接受前缀的状态在片上重建(与 ReplaySSM 独立同思路),单 fused kernel 覆盖 shortconv/归一化/门控/递归/输出归一化;Block AttnRes prefill 用 SP 物化块表示(TP all-reduce 拆成 reduce-scatter + all-gather,中间插 intra-block kernel),decoding 时 inter-block kernel 走 side stream、intra-block 融合进 TP all-reduce;LatentMoE 把 down-proj 与 router 融合成单 GEMM、latent 权重跨 rank 分片 + multimem store 把 all-gather 融进 epilogue;routed 专家 decoding 用 WarpDecode 式 token-centric kernel(每 warp 负责一个输出神经元、warp 内再分 lane team 分专家 + warp 级归约,权重布局离线置换减少运行时反量化开销)。
- **集群级调度**:cache-aware 亲和调度(一致性哈希把会话钉在主/备两个集群,故障时重 prefill 分摊到全集群);budget-based 准入控制(按请求类别分资源预算,防 1M 长请求饿死 <2K 短请求)。

---

## 5. 评测与案例(§6/§7)

- 评测覆盖推理知识(GPQA 93.5%、HLE-Full 56.0% w/ tools)、编程(ProgramBench 77.8% 第一、SWE-Marathon 42.0% 领先 Claude Fable 5 七分、Terminal-Bench 2.1 88.3% vs GPT-5.6 Sol 88.8%)、Agentic(BrowseComp 91.2%、DeepSearchQA 95.0 F1、MCPMark-Verified 94.5% 等 SOTA;GDPval-AA v2/AA-Briefcase 等 Elo 类由 Claude Fable 5 领先)、视觉(OmniDocBench 91.1% 第一、Math-Vision 94.3% → +Python 97.8%、ZeroBench +Python 41.0%)。整体仅次于 Claude Fable 5 / GPT-5.6 Sol。
- 案例:24 小时预算内优化 AttnRes/DSA/KDA/MLA 四个 kernel(KDA 运行时间降 73.6%);自主开发 MiniTriton 编译器(开源);48 小时用开源 EDA 完成一颗 nano 推理芯片 RTL 设计(nano-kpu,开源);2 小时复现 I-Love-Q 天体物理关系;Kimi Work 知识工作(87 份季报 + 99 份 PDF、2800+ 次搜索);视频剪辑/动效设计。
- 有趣的自举细节:后期开发中,大部分 kernel 优化工作已由一个早期 K3 checkpoint 自己完成。

---

## 6. 与 `~/source_code/` 源码的关联

- **KDA / Kimi Linear 推理实现**:`vllm/vllm/model_executor/models/kimi_k3.py`、`kimi_linear.py`(以及 `vllm/vllm/transformers_utils/configs/kimi_k3.py`、`kimi_linear.py`),测试见 `vllm/tests/models/kimi_k3/test_kda.py`、`test_kda_metadata.py`、`test_amd_kda_decode.py`,kernel benchmark 见 `vllm/benchmarks/kernels/benchmark_kimi_k3_kda_decode.py`;KDA 上下文并行测试见 `vllm/tests/distributed/test_kimi_linear_context_parallel.py`(对应 §5.1.2 KCP)。**SGLang** 侧对应 `sglang/python/sglang/srt/models/kimi_k3.py`、`kimi_linear.py`、`kimi_k3_vl.py` 及 `sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`、`linear/`(混合线性注意力 backend,对应 KDA+MLA 双缓存管理 §5.4.1)。KDA 的 chunkwise kernel 上游在 FLA(flash-linear-attention,论文 §5.1 提到 FlashKDA 作为其后端、KCP 实现于 FLA PR #691;本机未 clone FLA,如需深入可 clone 到 `~/source_code/`)。
- **MLA**:K3 的 Gated MLA 源自 DeepSeek MLA,参考实现见 vllm/sglang 的 DeepSeek 模型与 MLA backend;NoPE 全局层 + 线性层提供位置信息的做法与 `sglang`/`vllm` 中 hybrid linear attention 模型的实现一致。
- **MoE 训练/推理 kernel**:Stable LatentMoE 的 fused router+down-proj GEMM、token-centric decoding kernel(WarpDecode 式)思想与 `source_code/DeepGEMM/` 的 group GEMM 及 vllm/sglang 的 MoE kernel 相关;QAT 的 MXFP4 专家量化与 DeepGEMM 的 FP8/微缩放格式工作同族。
- **KV cache / 传输**:训练期激活跨 PP rank 远程 offload 用的是 **Mooncake Transfer Engine**(`source_code/mooncake/`);vllm 的 mooncake KV connector 见 `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/`。
- **Muon 优化器**:Per-Head Muon 是 Moonlight/Muon 系工作(论文引 `liu-2025-moonlight`),KTransformers 等仓库中有 Muon 相关集成可参考。
- **EAGLE-3 投机解码**:§4.1.4 的 MTP→EAGLE-3 draft 微调对应 vllm `kimi_k25_eagle3.py` / sglang `kimi_k25_eagle3.py` 及 `vllm/tests/models/kimi_k3/test_eagle3.py`。
- 模型 config 实证:`~/model-knowledge/configs/kimi-k3/config.json` 中可见 `activation_situ_beta: 4.0`、`activation_situ_linear_beta: 25.0`(SiTU-GLU 的 β₁/β₂)、`attn_res_block_size: 12`(Block AttnRes 块大小)、`latent_moe_use_norm: true`、`linear_attn_config.full_attn_layers`(MLA 层位置)等,与论文 §2 完全对应。
