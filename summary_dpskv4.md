# DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence

- arXiv: 2606.19348(TeX 源码缓存:`~/.cache/nanochat/knowledge/2606.19348/`,入口 `main.tex`)
- 机构:DeepSeek-AI
- 定位:DeepSeek-V4 系列 **preview 版**技术报告,包含 **DeepSeek-V4-Pro**(1.6T 总参数 / 49B 激活)与 **DeepSeek-V4-Flash**(284B 总参数 / 13B 激活)两个 MoE 模型,均原生支持 **100 万 token 上下文**。核心目标:打破超长上下文的效率瓶颈,为 test-time scaling 和长程 agent 任务奠基。

## 1. 模型架构(Section 2)

整体沿用 DeepSeek-V3 框架:Transformer + DeepSeekMoE + MTP(深度 1),其余未说明细节均同 V3。在此之上三大升级:

### 1.1 混合注意力:CSA + HCA(Section 2.3)

两种新型高效注意力交错堆叠,替代 V3 的 MLA:

- **CSA(Compressed Sparse Attention)**:先把每 m 个 token 的 KV 压缩成 1 条压缩 KV 项(序列长度缩到 1/m)。压缩方式:由 hidden state 算出两路 KV 项 C^a/C^b 与压缩权重 Z^a/Z^b,加可学习位置偏置后做 row-softmax 加权求和;C^a 与 C^b 的索引有重叠(每个压缩项实际来自 2m 个原始项),净压缩率仍是 1/m。然后在压缩 KV 上施加 **DeepSeek Sparse Attention(DSA,源自 V3.2)**:lightning indexer 用低秩方式产生 indexer query(与主注意力的 latent query c^Q 共享),ReLU 打分 + 可学习 head 权重,top-k 选出压缩块做 **Shared-KV MQA**(压缩 KV 项同时充当 key 和 value)。
- **HCA(Heavily Compressed Attention)**:更激进压缩,每 m′(≫ m)个 token 压成 1 条,无重叠、**不做稀疏选择**,压缩后直接 dense MQA。
- 公共设计:grouped output projection(c·n_h 很大,先分组投影到 d_g 再拼起来投影到 d);query/KV 各项在 core attention 前加 RMSNorm 防 logits 爆炸(因此 Muon 中不再需要 QK-Clip);部分 RoPE(只对每个向量的最后 64 维加 RoPE,并对 attention 输出加位置 -i 的 RoPE 使输出携带相对位置信息);额外滑动窗口分支(窗口 n_win=128 的未压缩 KV,补足块内因果与局部依赖);attention sink(每 head 可学习 sink logit)。
- 效率:KV cache 混合精度存储(RoPE 维度 BF16,其余 FP8),indexer 注意力用 FP4 计算,top-k 比 V3.2 更小。1M 上下文下,V4 系列 KV cache 仅为 BF16 GQA8/128 维基线的约 2%;V4-Pro 相对 V3.2 仅需 27% 单 token FLOPs 和 10% KV cache;V4-Flash 更极致(10% FLOPs / 7% KV cache)。

具体超参(§4.2.1):
- **V4-Flash**:43 层,d=4096;前 2 层纯 SWA;CSA m=4、indexer 64 头×128 维、top-k=512;HCA m′=128;n_h=64、c=512、d_c=1024、g=8、d_g=1024;每层 1 shared + 256 routed experts(expert 中间维 2048),每 token 激活 6 个;前 3 个 MoE 层用 Hash routing。总 284B / 激活 13B。
- **V4-Pro**:61 层,d=7168;前 2 层 HCA;top-k=1024;n_h=128、d_c=1536、g=16;每层 1 shared + 384 routed experts(expert 中间维 3072),激活 6 个。总 1.6T / 激活 49B。

### 1.2 mHC(Manifold-Constrained Hyper-Connections,Section 2.2)

强化残差连接:残差流宽度扩展 n_hc=4 倍(Hyper-Connections),但把残差映射矩阵 B_l 约束在双随机矩阵流形(Birkhoff polytope)上——谱范数 ≤1、非扩张、对乘法封闭,保证深层堆叠稳定。约束通过 Sinkhorn-Knopp 迭代(exp 后行列交替归一化,t_max=20)实现;输入/输出映射用 Sigmoid 保证非负有界。三个映射参数均由输入动态生成 + 静态偏置,门控因子 α 小值初始化。

### 1.3 MoE 细节调整

- 路由 affinity 打分从 Sigmoid 改为 **Sqrt(Softplus)**;仍用 auxiliary-loss-free 负载均衡 + 轻微 sequence-wise 平衡损失(权重 0.0001)防单序列内极端不均衡。
- 取消路由目标节点数(V3 的 node-limited routing)限制,重新设计并行策略。
- 最初几层 dense FFN 替换为 **Hash routing MoE 层**(按 token ID 的哈希定专家)。

### 1.4 Muon 优化器(Section 2.4)

大部分参数用 Muon(embedding、prediction head、mHC 静态偏置/门控、所有 RMSNorm 权重仍用 AdamW)。要点:Nesterov、weight decay 0.1、update RMS 缩放到 0.18 以复用 AdamW 学习率;**混合 Newton-Schulz 迭代**(共 10 步:前 8 步系数 (3.4445, -4.7750, 2.0315) 快速收敛,后 2 步 (2, -1.5, 0.5) 精稳在奇异值 1);因架构自带 QK RMSNorm,不用 QK-Clip。

## 2. 预训练(Section 4)

### 2.1 数据构建(§4.1 Data Construction)

- 在 V3 预训练数据基础上构建更多样、更高质量、有效上下文更长的语料。**预训练语料总量超过 32T tokens**,含数学、代码、网页、长文档等高质量类别。
- 网页数据:过滤批量自动生成/模板化内容,防 model collapse。
- 数学与代码仍是核心;**mid-training 阶段加入 agentic 数据**增强代码能力(未给 token 数)。
- 多语料:更大规模,覆盖长尾知识。
- 重点强调长文档(科学论文、技术报告等)。
- 预处理大体沿用 V3:tokenizer 在 V3 基础上加少量上下文构建特殊 token,词表仍 128K;保留 token-splitting 与 FIM;跨来源文档 packing 减少截断;**与 V3 不同,预训练采用 sample-level attention masking**。

### 2.2 训练设置(§4.2.2 Training Setups)

- **训练 token 数:V4-Flash 训练 32T tokens;V4-Pro 训练 33T tokens**(§1 与 §4.2.2 均明确)。
- 优化器超参:AdamW(β1=0.9, β2=0.95, ε=1e-20, wd=0.1);Muon momentum=0.95, wd=0.1。
- Batch size 调度:Flash 逐步增至 75.5M tokens 后保持;Pro 最大 94.4M tokens。
- 学习率:2000 步线性 warmup;Flash 峰值 2.7e-4、余弦衰减至 2.7e-5;Pro 峰值 2.0e-4、衰减至 2.0e-5。
- 序列长度课程:4K → 16K → 64K → 1M 逐步扩展。
- 稀疏注意力引入:Flash 前 **1T tokens 用 dense attention  warmup**,在 64K 序列长度阶段引入稀疏注意力,先短阶段 warmup lightning indexer,之后全程稀疏;Pro 的 dense 阶段更长(具体未披露),引入策略相同(两阶段)。
- MTP loss 权重:大部分训练 0.3,学习率开始衰减后降为 0.1。
- **论文未披露**:各序列长度阶段的 token 数、mid-training/annealing 的 token 规模、epoch 数、数据重复使用情况(语料 "超过 32T" 与 Flash 训 32T / Pro 训 33T 的关系未明示,是否多 epoch/高质量子集重复均未说明)。

### 2.3 训练稳定性(§4.2.3)

万亿参数 MoE 训练出现 loss spike,根源定位到 MoE 层 outlier 与路由机制的恶性循环。两个实用对策:

- **Anticipatory Routing(预见性路由)**:第 t 步用当前参数 θ_t 算特征,但路由索引用历史参数 θ_{t-Δt} 计算——提前取数、预计算并缓存路由索引,解耦主干与路由网络的同步更新。infra 上把额外开销压到约 20% wall-clock;并有自动检测机制:只在 loss spike 时短暂回滚并启用该模式,过后恢复常规训练。
- **SwiGLU Clamping**:线性分量 clamp 到 [-10, 10],gate 分量上限 cap 10(借鉴 gpt-oss),消除 outlier,不损性能。

### 2.4 Base 模型评测

V4-Flash-Base 以更小参数多数基准超过 V3.2-Base;V4-Pro-Base 全面最强(知识、长上下文提升尤其大,如 FACTS Parametric 62.6 vs V3.2 的 27.1,Simple-QA verified 55.2 vs 28.3)。

## 3. 后训练(Section 5)

两阶段范式:**领域专家独立培养 → on-policy 蒸馏统一合并**(替代 V3.2 的混合 RL 阶段)。

- **Specialist Training(§5.1.1)**:每个领域(数学、代码、agent、指令遵循等)单独训练专家:先领域 SFT,再 GRPO RL(超参对齐 R1/V3.2)。
  - 三种推理力度模式:Non-think / Think High / Think Max(RL 中用不同长度惩罚与上下文窗口;Max 模式在 system prompt 前注入专用指令;评测上下文窗口分别 8K/128K/384K)。
  - **Generative Reward Model(GRM)**:难验证任务不再用标量 RM,改用 rubric 引导数据 + GRM 评判;actor 网络原生充当 GRM,生成与评判能力联合优化,只需少量人类标注。
  - 新 tool-call schema:XML 格式 + 特殊 `|DSML|` token,减少转义错误。
  - Interleaved Thinking 升级:工具调用场景全程保留推理轨迹(跨 user 轮次),普通对话仍按 V3.2 策略丢弃。
  - Quick Instruction:把辅助任务(是否搜索、生成标题/query、权威性/领域分类、URL 是否读取)做成专用 special token,复用已算好的 KV cache,免掉小模型重复 prefill,降低 TTFT。
- **On-Policy Distillation(OPD,§5.1.2)**:十余个领域教师模型,学生在自采轨迹上最小化与教师的 **reverse KL(全词表 logits 级)**,按任务上下文选择性对齐相应专家;全词表蒸馏比 token 级 KL 估计方差更低、更稳。

## 4. Infra / 系统工程(Section 3、Section 5.2)

- **细粒度 EP 通信-计算重叠(§3.1)**:MoE 层分 Dispatch/Linear-1/Linear-2/Combine 四阶段,通信总量 < 计算总量,把专家切成 wave 做细粒度流水(比 Comet 更细),单 kernel 融合通信+计算。对 NVIDIA GPU 和华为昇腾 NPU 均验证:通用推理 1.50~1.73× 加速,RL rollout 等长尾小 batch 场景最高 1.96×。**已开源为 DeepGEMM 组件 MegaMoE**(github.com/deepseek-ai/DeepGEMM/pull/304 —— 对应本机 `~/source_code/DeepGEMM/`)。还给硬件厂商四点建议:关注计算/通信比(V4-Pro 下每 GBps 带宽可藏 6.1 TFLOP/s 计算)、功耗余量、更低延迟的跨 GPU 信号(push 通信)、用无指数/除法的廉价逐元素激活替代 SwiGLU。
- **TileLang kernel 开发(§3.2)**:用 TileLang DSL 替代数百个 ATen 算子;Host Codegen 把 host 端检查从 Python 挪到生成代码(每次调用开销从数十~数百微秒降到 <1µs);集成 Z3 SMT solver 做形式化整数分析;默认关 fast-math、提供 IEEE intrinsics,支持与手写 CUDA 逐 bit 对齐。
- **Batch-invariant 与确定性 kernel(§3.3)**:训推 bitwise 一致。Attention 放弃 split-KV,改用双 kernel 策略(整序列单 SM + 末波多 SM 同累加顺序,distributed shared memory);矩阵乘法全面用 **DeepGEMM** 替代 cuBLAS,不用 split-k 但性能持平;稀疏注意力 backward 用每 SM 独立 buffer + 确定性全局求和替代 atomicAdd;MoE backward 用 rank 内 token 预排序 + 跨 rank buffer 隔离。
- **训练框架(§3.4)**:
  - Muon × ZeRO 混合策略:dense 参数限制 ZeRO 并行度 + 背包算法分桶(padding <10% 开销);MoE 参数按专家展平后均分;同形状参数合并以批量跑 Newton-Schulz;NS 迭代可用 BF16;MoE 梯度随机舍入量化到 BF16 再同步(通信减半),reduce-scatter 改 all-to-all + 本地 FP32 求和两阶段。
  - mHC 高效实现:融合 kernel + 选择性重计算 + 调整 DualPipe 1F1B,wall-time 开销仅 6.7%。
  - 压缩注意力的两阶段 Context Parallelism:先相邻 rank 传最后 m 条未压缩 KV 解决跨界压缩,再 all-gather + fused select-and-pad 重组。
  - 基于 TorchFX 的 tensor 级自动激活检查点:标注单 tensor,自动回溯最小重计算子图,零拷贝释放/复用显存,自动去重共享存储的 tensor。
- **推理框架(§3.5)**:混合注意力破坏 PagedAttention 假设,定制 KV cache 布局——经典 KV cache(CSA/HCA 压缩块,每块覆盖 lcm(m,m′) 个原始 token)+ state cache(SWA 与未够 m 个的待压缩尾部 token,按定长块分配,视作 state-space 模型状态)。**On-disk KV cache** 用于共享前缀复用:压缩 KV 直接落盘;SWA KV(体积约压缩 KV 的 8 倍)三策略:全量缓存 / 周期 checkpoint(每 p token 存最后 n_win 个)/ 零缓存(重算最后 n_win·L 个 token 恢复)。
- **后训练 infra(§5.2)**:
  - **FP4(MXFP4)QAT**:MoE 专家权重 + CSA indexer QK 路径;FP4→FP8 反量化无损(FP8 E4M3 动态范围可吸收 1×32 子块 scale),QAT 直接复用 FP8 训练框架,反向等价 STE;index score FP32→BF16 使 top-k 选择提速 2× 且保持 99.7% 召回。
  - 全词表 OPD 教师调度:教师权重集中分布式存储按需加载,只缓存教师末层 hidden state(用时过 prediction head 重算 logits),按教师索引排序样本使每个 mini-batch 只加载一个教师头,专用 TileLang kernel 算精确 KL。
  - 可抢占容错 rollout:token 级 WAL + KV cache 保存/恢复;指出被打断请求从头重生会引入长度偏差(数学上不正确)。
  - 百万 token RL:rollout 数据拆轻量元数据 + 重 per-token 字段(共享内存加载、mini-batch 粒度即释放)。
  - **DSec 沙箱平台**(Rust 三组件 + 3FS):统一接口下四种执行底座(函数调用/容器/microVM/fullVM),EROFS/overlaybd 分层按需加载,单集群数十万并发沙箱,轨迹日志支持抢占后快进恢复。

## 5. 评测要点

- 知识:SimpleQA-Verified 超所有开源模型 20 个点,仍逊 Gemini-3.1-Pro。
- 推理:V4-Pro-Max 超 GPT-5.2/Gemini-3.0-Pro,略逊 GPT-5.4/Gemini-3.1-Pro(约落后前沿 3~6 个月);Codeforces 与 GPT-5.4 相当(人类榜约第 23 名);Putnam-2025 形式化数学 120/120。
- Agent:公开 benchmark 与 Kimi-K2.6、GLM-5.1 持平;内部评测超 Sonnet 4.5、接近 Opus 4.5;Terminal-Bench 2.0 Verified 约 72.0。
- 长上下文:1M token MRCR 超 Gemini-3.1-Pro(逊 Opus 4.6),CorpusQA 也更好;128K 内检索稳定,之后缓慢衰减。
- 真实任务:中文写作对 Gemini-3.1-Pro 胜率 62.7%;白领任务对 Opus-4.6-Max 非负率 63%;内部代码 agent 任务接近 Opus 4.5;85 人内部调查 52% 愿意设为默认编码模型。

## 6. 与本机源码仓库的关联

- **DeepGEMM**(`~/source_code/DeepGEMM/`):论文两大开源落点都在这里——MegaMoE 融合 EP kernel(PR #304)和 batch-invariant 矩阵乘法(端到端替代 cuBLAS)。阅读该仓库可对照 §3.1/§3.3。
- **vllm / sglang**(`~/source_code/vllm/`, `~/source_code/sglang/`):V4 的混合 CSA/HCA 注意力 + 异构 KV cache(state cache + 压缩块 cache)对推理框架的 PagedAttention 假设的破坏(§3.5),是理解这两个引擎中 hybrid KV cache 管理、SWA 支持和 DSA/sparse attention 实现(vLLM 已有 DeepSeek V3.2 DSA 的 indexer/sparse 路径,sglang 同样)的钥匙;on-disk KV cache 与 prefix reuse 对应 vLLM 的 prefix caching / KV offloading 机制。
- **ktransformers**(`~/source_code/ktransformers/`):CPU/GPU 异构 MoE 推理,可对照 V4 MoE(256/384 细粒度专家 + Hash routing)在资源受限场景的 offload 实现。
- **mooncake**(`~/source_code/mooncake/`):分布式 KV cache/传输,对应 §3.5 的 on-disk/分布式 KV 存储与共享前缀复用,以及 §5.2.3 rollout 中 KV cache 保存/恢复的场景。
- **pytorch**(`~/source_code/pytorch/`):§3.4.4 基于 TorchFX 的 tensor 级激活检查点是对 PyTorch autograd 的扩展,可对照 `torch.utils.checkpoint` 与 TorchFX tracer。
- **cache-dit**(`~/source_code/cache-dit/`):Diffusion Transformer 的 attention cache,与 CSA/HCA 的"压缩-复用 KV"思想在不同模态上的呼应(见 `~/knowledge/summary_cache_dit.md`)。

## 7. 局限与未来方向

架构偏复杂(为降风险堆叠了多个已验证组件),未来做减法;Anticipatory Routing 与 SwiGLU Clamping 的机理不明;探索 embedding 等更多维度稀疏化、多模态、数据合成策略。
