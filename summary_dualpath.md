# DualPath 论文总结(arXiv 2602.21548)

> 来源:`~/.cache/nanochat/knowledge/2602.21548/`(arXiv TeX 源码,入口 `paper.tex`,正文在 `session/01`–`11`)。
> 论文标题:*DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference*(北大 + 清华 + DeepSeek-AI,2026-02)。
> 一句话:agentic 多轮推理的瓶颈已从算力变成 **KV-Cache 存储 I/O**;PD 分离架构下 prefill 引擎(PE)的存储网卡(SNIC)被打满而 decode 引擎(DE)的 SNIC 闲置。DualPath 新增 **storage→DE→RDMA→PE** 第二条加载路径,把全集群 SNIC 带宽池化调度,离线吞吐最高 **1.87×**、在线服务能力平均 **1.96×**。

## 1. 背景与动机(§2–§3)

- **Agentic 负载特征**:长上下文、短 append、多轮交互。生产 trace(编程 agent):平均 157 轮,平均 context 32.7K,平均 append 仅 429 token → **KV-Cache 命中率 98.7%**(一般 ≥95%)。每轮绝大多数 token 命中缓存,只有新增 token 需要真正 prefill 计算。
- **I/O 成为主导**:用 cache-compute ratio(GB/PFLOP,append=429,context 16K–64K)量化:Qwen2.5-32B(FP16)117–267、GPT-OSS-120B 47–95、Qwen3-235B 39–60、DS-V3.2 660B 13–36、DS-V3 4.8–5.8。MLA/sparse attention 已经把 KV-Cache 压得极小,其他模型更糟。
- **硬件趋势不利**:从 Ampere 到 Blackwell,GPU 的 I/O-compute 比下降 14.4×(FLOPS 涨得比网卡带宽/HBM 容量快)。
- **核心结构性失衡**:PD 分离 + 外部 KV-Cache 存储(SSD,如 3FS)架构下,命中 KV-Cache **只由 PE 从存储加载** → PE 侧 SNIC 持续打满,DE 侧 SNIC 完全闲置。给 PE 单方面加带宽又贵又不通用。
- 顺带:RL 训练的 rollout 阶段同一负载形态,且 DRAM 被训练状态(optimizer/reward model offload)占满,**Mooncake 式 DRAM pool 在此不适用**;在线服务大 working set 下 DRAM pool 也不经济 → 直接面向 SSD 存储做负载均衡才是正解。

## 2. 系统设计(§4)

沿用两个主流技术:**PD 分离** 与 **layerwise prefill**(LayerKV / PrefillOnly:每层只加载本层 KV-Cache,HBM 只需容纳一层,等效 batch 扩大 ≈ 层数倍)。

### 2.1 双路径加载

每引擎管理一张 GPU;每 PE/DE 配少量 DRAM 作 **PE buffer / DE buffer**。

- **PE 读路径(传统)**:hit KV-Cache 从存储读入 PE buffer → 逐层 H2D 到 PE HBM 参与注意力计算 → 全量(hit+miss)KV-Cache 逐层传给 DE buffer。全程与计算 overlap,重复 n_layer 次。
- **DE 读路径(新增)**:hit KV-Cache 先读入 **DE buffer** → prefill 时逐层经 **CNIC RDMA 走计算网络** 转发到 PE HBM(与计算 overlap);算完后只需把 miss token 的新 KV-Cache 传回 DE buffer 合并。
- **Decode 阶段**:DE 收齐完整 prompt KV-Cache 后分配 HBM、H2D、释放 host 内存再解码;每攒满一个 block(如 64 token)立即落盘持久化。
- **为什么留 DE buffer 而不是 GPUDirect RDMA 直通**:省 HBM;且该场景生成长度短、TTFT 占比不可忽视。
- **Block 布局**(附录 D):**Full Block** `[layer, tokens, bytes]` 用于所有存储交互(trie 组织);**Layer Block** `[1, tokens, bytes]` 用于 PE↔HBM↔DE 的逐层流式传输;n 个 Layer Block 直接拼接即得 Full Block,免 layout 转换。

### 2.2 无瓶颈分析(§4.2)

记号:每节点 g 张 GPU、每 GPU 一个带宽 B 的 CNIC;整机存储带宽 s·B;内存带宽 M。逐链路推导 PE/DE 的 CNIC 读写与 DRAM 压力,得到不引入新瓶颈的 P/D 配比区间:

```
s/(g−s) ≤ P/D ≤ min{ (g−2s)/s, (g−s)/(2s), (M/Bs − 3)/2 }
```

对 g=8、s=1、M≈500GB/s、Bs≈50GB/s:**1/7 ≤ P/D ≤ 7/2**,覆盖绝大多数实际部署。

### 2.3 CNIC 为中心的流量管理(§5)

问题:GPUDirect Storage / CUDA copy engine 等传输路径与计算网络 QoS 体系不共享,GPU 又不支持 PCIe QoS;而 EP AllToAll 等集合通信是亚毫秒级突发,软件 traffic shaper 来不及插空 → KV 流量会干扰延迟敏感的推理通信。

方案:**所有进出 GPU 的流量(含本地 H2D/D2H)一律走配对 CNIC 的 GPUDirect RDMA 路径**,统一纳入计算网络原生 QoS:
- InfiniBand 用 **Virtual Lane** 分级:推理通信走高优先级 VL,KV-Cache 走低优先级 VL,交换机/网卡 WRR 仲裁把 ~99% 带宽留给高优 VL(低优保留少量防饿死)。附录 A 给出具体 `qos_*` 配置;RoCE 侧对应 DSCP→TC + PFC + 加权调度。
- **CNIC 辅助 H2D/D2H**:H2D = 先读进 DRAM,再向本 GPU 配对 CNIC 提交本地 RDMA Write。看似绕路,但这是目前唯一能让 KV 传输不劣化集合通信的实用方法。
- 附带收益:大量小块传输时 CNIC 路径比 copy engine 更快——`cudaMemcpyAsync` 提交一次 ~5–7µs(CUDA driver 闭源无法进一步拆解),而 RDMA Write WR 提交只是用户态几次 mmio 写、~1µs,还可 doorbell batching 摊销。

### 2.4 调度(§6)

两级调度,目标同时平衡 **NIC 流量** 与 **GPU 利用率**;引擎分组(组内全 PE 或全 DE,同节点同组),只有 Leader Engine 与中央调度器交互。每个引擎上报:未完请求数 `seq_e`、未完 token 数 `tok_e`、所在节点磁盘读队列 `read_q`。token 数是 GPU/磁盘/网络负载的统一代理。

- **PE 调度(全局 FIFO)**:引擎分三类——超载(`tok_e>β`)、读队列短(`read_q≤α` 且未超载)、读队列长。优先从"读队列短"类里选 `tok_e` 最小者(防止 SNIC 吃不饱),否则从第三类选。α = 3 秒可读的 token 数,β = 一张 GPU 5 秒能处理的 token 数(预先 profile)。
- **DE 调度(两级,不保全局 FIFO)**:先按组总 token 数最小把全局队列分配到组私有队列;组内按剩余 HBM 求可调度上界集合 R,算高水位阈值 `Z = 1.05×(ΣR len + Σ tok)/|E|`,优先把请求派给加了它也不超 Z 的 DE 中 `seq_e` 最小者,其次才动高超载 DE(选 tok 最小,降低 HBM 耗尽/抢占风险)。
- **读路径选择**:请求定了 (PE, DE) 后,**谁的读队列短就从谁那边读**。请求级双路拆分留作 future work。
- **Intra-engine(仅 PE)**:attention 常用 DP,各 GPU 服务不同请求集合,attention 结束后必须同步进 FFN → 需要对齐各 GPU 的 attention 层执行时间。做法:按 (cached, bsz) 对估算注意力理论计算量(时间关系预先 profile 拟合),FIFO 装箱直到超过 **compute quota**(实验设 300ms);超限时对 bsz 二分查找做 chunked prefill 塞满剩余配额。效果:EP 组内 attention 执行时间 Max/Avg 低至 1.06。

## 3. 实现与评测(§8)

- 实现:自研推理框架上改 ~5K 行;kernel 栈 FlashMLA + DeepGEMM + DeepEP(与主流开源一致);存储用 **3FS**(io_uring 风格 kernel-bypass 接口,集群 3FS 无内部 DRAM cache、能打满 400Gbps SNIC)。
- 测试床:8×Hopper/节点,8×400Gbps CNIC(IB)+ 1×400Gbps SNIC,计算/存储网络物理隔离。
- 模型:DS 660B(DeepSeek-V3.2,MLA+DSA MoE)、内部 27B 缩比版(hidden 2560/30 层/72 routed experts,附录 B)、Qwen2.5-32B(GQA dense)。
- 数据:三个 agent trace(各 500 条轨迹,MaxLen 32K/48K/64K,Turns 60/106/157)。
- 基线:**SGL(MC)** = SGLang(commit 19089aa)+ HiCache + Mooncake Store(3FS 后端)+ Mooncake Transfer Engine;**Basic** = 未改的自研框架;**Oracle** = 去掉全部 I/O 的理论上界。

### 主要结果

- **离线(RL rollout 形态)**:DS 660B(2P4D)最高 **1.87×** vs Basic,且贴平 Oracle;DS 27B 最高 1.78×;append/生成长度越短优势越大(append 放大时 Basic 逐渐追平——瓶颈切回 GPU 计算),不同 append 缩放下加速 1.82–1.99×。
- **P/D 配比实验**(DS 27B,1P1D/2P1D/1P2D):DualPath 平均 1.64×、最高 2.46×;关键证据——"Basic 1P1D ≈ Basic 1P2D、DualPath 1P1D ≈ Basic 2P1D、DualPath 2P1D ≈ DualPath 1P2D",每对系统等效可用存储带宽相同,**直接证实存储带宽是主导瓶颈**。
- **在线(Poisson 到达,SLO: TTFT≤4s、TPOT≤50ms)**:APS 容量 DS 27B 1.67×、DS 660B 2.25×(平均 **1.96×**);TTST/TPOT 不退化,Basic 的 TTFT 排队时间因带宽不足爆炸而 DualPath 各成分稳定。
- **消融**(DS 660B,64K,1024/2048 agents):layerwise prefill −17.2% JCT;再加 dual-path −38.2%;再加调度 −45.6%。
- **负载均衡**:SNIC 流量 Max/Avg 从 round-robin 的 1.53 降到 1.18。
- **大规模**:最大 1152 GPU(48P96D、48K agents)离线 JCT 与 2P4D/2K agents 几乎相同(3201s vs 3167s,近线性);在线 44P88D 达 22× APS 且延迟持平;调度器 CPU <10 核,不是瓶颈。

### Discussion(§9)

- **Working set 估算**:`λ·T̄·total_len/2`。DS 660B serving 下 69GB(APS 0.1)→681GB(APS 0.45)。真实场景有 tool 延迟等间隙:JCT 拉长 r 倍 → APS 容量涨 r 倍 → working set 涨 **r²**(成本 r³),会超出内存池容量、降低命中率——所以绕过 DRAM 直打 SSD 的设计在真实 agentic 场景更有必要。
- Future work:请求级双路径拆分读、更自适应的并行度/P-D 配比(模拟器或在线调整)、大规模下更低的 TTFT 分位数。

## 4. 与 `~/source_code/` 源码的联系

- **SGLang(`source_code/sglang/`)**:论文基线 SGL(MC) 的全部组件都在——`python/sglang/srt/disaggregation/`(PD 分离,prefill.py/decode.py + mooncake/nixl/mori 等多种 conn)、HiCache(`srt/managers/cache_controller.py` HiCacheController)、以及 decode 侧 KV offload(`disaggregation/decode_kvcache_offload_manager.py`)。**DualPath 的 DE 读路径相当于给这套 disaggregation 体系加一条"存储→DE→PE"反向 KV 转发通路**,调度上把 SNIC 队列长度纳入派单。
- **Mooncake(`source_code/mooncake/`)**:`mooncake-transfer-engine/` 是论文中 PD 间 KV 传输基线;`mooncake-store/` 是 DRAM pool 方案。DualPath 的论点正是:DRAM pool 在 RL rollout(DRAM 被训练状态占用)和大 working set 在线服务下不够用/不经济,应直接面向 SSD 存储并把 **所有引擎的 SNIC 带宽池化**——可与 Mooncake 分层组合(论文称叠加 DRAM 中间层收益 marginal)。
- **vLLM(`source_code/vllm/`)**:`vllm/distributed/kv_transfer/kv_connector/v1/` 下的 connector 抽象(含 `nixl/connector.py`)是 PE↔DE KV 传输的挂载点;DualPath 思路可落地为一个新 connector:命中 KV-Cache 由 DE 侧从存储拉取再经 NIXL/RDMA 转发给 PE。
- **NIXL(`source_code/nixl/`)**:NVIDIA 的传输抽象层,天然适合做 DE→PE 的 KV 转发通道。
- **DeepGEMM / FlashMLA / DeepEP**:论文实现 kernel 组合 = `source_code/DeepGEMM` + FlashMLA + DeepEP(EP AllToAll 正是 §5 要保护的延迟敏感流量)。
- **3FS**:DeepSeek 的 Fire-Flyer File System,论文存储后端(本地无仓库)。

## 5. 启发与可借鉴点

1. **瓶颈再定位**:agentic 负载(≥95% 命中率、短 append、上百轮)下"优化数据搬运 > 优化计算"。cache-compute ratio(GB/PFLOP)是个好用的量化工具,配合本地 `model-knowledge/configs/` 可快速估算任意模型的 I/O 压力。
2. **闲置资源池化**:不发明新硬件,把 DE 侧闲置 SNIC + 计算网络空闲带宽变成可调度资源。"Basic 1P1D ≈ DualPath 1P1D 加一路径 ≈ Basic 2P1D"这类等效带宽对照实验是论证 bottleneck 的范式。
3. **QoS 收敛到 CNIC**:把所有 GPU 进出流量(含本地 H2D)强制走 CNIC RDMA,用 IB VL / RoCE TC 做硬隔离(高优 ~99% 带宽);小块传输场景 RDMA WR 提交(~1µs)还比 cudaMemcpyAsync(5–7µs)便宜。
4. **调度即多维负载均衡**:GPU token 数 + 节点磁盘读队列联合派单;attention-DP 下用 compute quota 对齐各卡 attention 层耗时,消除同步气泡。
5. **Layer/Full 双 block 布局**:存储交互用 Full Block、层间流式用 Layer Block,免 layout 转换,直接可拼。
