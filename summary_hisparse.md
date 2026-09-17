# HiSparse: Scaling Sparse-Attention Decoding with Hierarchical KV Cache Management

- **arXiv**: 2608.07009(NeurIPS 2026 preprint;Stanford & Meta / Alibaba / Ant Group / SJTU / PKU;Zhiqiang Xie 通讯)
- **状态**: 已 merge 进上游 SGLang(`--enable-hisparse` 一个开关启用;~2200 行 Python + CUDA kernel;致谢 Alibaba Cloud TairKVCache、蚂蚁 SCT、百度 Baige、智谱)
- **TeX 源码缓存**: `~/.cache/nanochat/knowledge/2608.07009/`
- **结合阅读**: `summary_dpskv32.md`(DSA indexer)、`summary_hysparse.md`(跨层共享 KV/选择,与 GLM-5.2 IndexShare 同思路)、`discussion_kv_selection_signals.md`(选择信号三部曲)、`summary_declarative_attention.md` / `summary_cacheblend.md`(分层 KV 的另外两个切面)

## 一句话总结

Top-k sparse attention(DSA/NSA/Quest)把每步 decode 的 attention **读**降到 k(~2048)个 KV entry,但 serving 系统仍把**完整上下文**的 KV 全部驻留 HBM(因为"现在没选中的位置以后可能被选中"),于是 decode 先撞**容量墙**而非算力墙:128K token 的 GLM-5.1 请求要 13.09 GB KV,1M token 请求直接占满一张 H200。HiSparse 把"逻辑上可得"与"物理上驻留"解耦:全量 KV 放 pinned host DRAM 作权威副本,GPU 上每请求每层只保留固定大小 B(实用区间 2k–4k)的 LRU GPU cache;每层 attention 前由一个融进 decode CUDA graph 的 **Resolve 融合 kernel** 完成命中检测/LRU 替换/host 抓取(借 Strata 的 GPU-assisted IO:`ld.global.nc.v2.b64` 向量化非一致 load);对 IndexShare 类跨层共享选择模型(GLM-5.2:21 个 anchor 层跑 indexer,57 个 shared 层复用)做**精确 layer-wise prefetch** 把残余 miss IO 与中间层计算重叠。**输出逐 bit 不变、indexer 无关**,长上下文峰值生成吞吐最高提升 4.7×,no-IO oracle 证明 resolve 机制本身零开销——唯一的代价就是 host↔device IO。

## 问题分解

1. **容量墙的形式化**: 约束是 `N_batch × L_ctx × KV/token` 必须塞进 weights 之外的 HBM,而每步只读 `N_batch × k`。GLM-5.1 部署实测(8×H200, 1.1TB):32K 输入时每请求 ~4GB,~60 并发打满;128K 只剩 ~15 并发。PD-colocated 下 decode KV 挤占 prefill 内存 → TTFT 随负载飙升;PD-disaggregated 下直接封死 decode pool 吞吐。
2. **朴素 offload 不可行**: 若每层每步都从 host 取 k 条 KV,GLM-5.1 每生成一个 token 要搬 ~200MB(78 层 × k=2048 × ~100KB/token),TPOT 30ms 下约 7GB/s/请求,十几个并发就榨干 PCIe Gen5 x16。
3. **出路在选择局部性**: indexer 在每层每步**提前**宣布要读哪些位置(精确的 demand signal);相邻步选择集高度重叠(连续步 reselect 大量相同位置),相邻层选择相关,新模型(GLM-5.2 IndexShare、IndexCache)干脆跨层共享 indexer 输出。→ sparse selection 像带强时间局部性的内存访问,正适合"小快缓存 + 大慢后端"的两级结构。

## 方法

### 设计不变量(invariants)

- **Complete KV availability**: 每请求全量 KV 在 decode GPU 之外有权威副本;
- **Bounded device footprint**: 每请求每层固定 B 个 slot(B ≥ k,与 L_ctx 无关),HBM 占用 `N_batch × N_l × B × W_KV × s`;
- **Exact outputs**: 只改 KV 摆放位置,不改选择/分数/输出;
- **Indexer-agnostic**: 只消费每层 emit 的 logical position 集合,不碰 DSA/NSA/Quest 内部;
- **Miss latency off critical path**。

### 两级层次与元数据(§3.2)

- **Host KV pool**: pinned host DRAM,权威全量副本。colocated 下 prefill 本地写入;disaggregated 下 prefill 通过 PD 传输路径直接写入 decode 实例的 host pool。
- **GPU cache**: 每请求每层 B slot;当前步选中的 k 条必须在内,剩下 B−k 个 slot 存"最近有用"的热条目。页表(page table)把 logical position 映射到 slot 或 host-only 哨兵;LRU 元数据 GPU 常驻。
- **不搬的东西**: indexer 选择状态(DSA per-token indexer key、NSA block key、Quest page min/max 摘要)永远 GPU 常驻——每 token 只有几百字节 vs KV ~100KB,小 2–3 个数量级。只有 attention 要读的 KV record 走层次结构,indexer 计算完全不动。
- **替换策略**: 每层独立 LRU,一个精修:步内 **hit 提升在新 fetch 的 miss 之上**(反复被选中的条目优先于一次性选择);per-layer 独立管理即可,跨层结构用 prefetch 利用而非协调驻留。
- **B 的选取**: 吞吐-延迟-HBM 三方折中,实用区间 **B ∈ [2k, 4k]**;host 链越快越倾向 B=2k(GH200 实测把最优点推向小 cache)。

### 请求生命周期(§3.3)

1. **Prefill & staging**: 每层 KV 产出即写 host pool,indexer 状态留 GPU;
2. **Admission**: 按 `N_l × B × W_KV × s` 预留(GLM-5.1 B=4096 时 ~0.4GB/请求 vs 128K 全量 13.09GB,~30×);
3. **Layer decode**: indexer 出选择集 → Resolve 查命中、取 miss、更新页表+LRU、输出与选择位置对齐的**物理 slot 稠密向量** → sparse attention gather 直接读;
4. **Write-through**: 新 token 的 KV 直接写进 GPU cache 保留 slot(最新位置必然驻留),backup stream 异步写穿 host pool,event 排序保证后续 fetch 可见。

### 融合 Resolve kernel(§3.4,CUDA graph 内,每层一次,每 block 一个请求)

像一个软件管理的 TLB:logical index 进、physical slot 出。五个 phase:
1. **Stage**: 选中位置装入 shared-memory hash 表;
2. **Mark**: 并行探查 B 个 slot 在不在选择集,标 hit / evictable;
3. **Scan**: 对 mark 做 parallel scan,压实 evictable、选够 victim、原地更新 LRU(hits 提到 MRU 端,victim 分给 miss,miss 排在 hit 之后);
4. **Fetch**: miss 的线程用 **GPU-assisted IO**(Strata 技术,`ld.global.nc.v2.b64` 向量化非一致 load,容忍 miss 的散乱地址,PCIe 和 NVLink-C2C 都省 transaction 开销)从 pinned host pool 直接搬;per-thread 传输块大小调优到接近链路带宽;
5. **Publish**: 更新页表,输出 `top_k_device_locs`。

### Layer-wise 精确 prefetch(§3.5)

- **针对 IndexShare/IndexCache 模型**: anchor 层一旦 emit 选择集,组内所有 shared 层的选择**已经确定**(提前若干层已知)。**plan-then-IO**: anchor 的 Resolve 额外记录 miss plan(哪条 host record 进哪个 slot),side stream 上的 copy-only kernel 把同样的 plan 重放进每个 shared 层的 cache,与中间层计算重叠。shared 层 cache 与 anchor 的 slot 布局 lockstep 一致,直接复用 anchor 的 slot 表:**等 prefetch 完成事件、完全跳过 resolution**(无 probe、无 LRU 更新、无同步 host load、零投机流量)。
- **投机变体(负结果)**: 对无共享索引的模型,用第 l 层选择hint 第 l+1 层——hit rate 仅边际提升,端到端无收益:hint 的位置 LRU 多半已驻留,而真正 miss 的(新进入 top-k 的位置)恰是 hint 预测不了的。结论:藏 miss 延迟需要**提前确定**选择,投机不可靠,**模型 co-design(跨层共享)才是正路**。

## 实验(§4)

**设置**: 三族 selector——GLM-5.1-FP8(DSA, token-level k=2048)、DeepSeek-V4-Flash(NSA 风格, top-512 over 4-token 压缩 KV entry,覆盖 2048 token)、Qwen3-30B-A3B + Quest(training-free, page 级);平台 8×H200(2TB host DRAM,最大工作点 host pool ~1TB pinned)、2×B200、GH200(NVLink-C2C)。baseline = 未改 SGLang v0.5.11 全量 HBM。SGLang `bench_serving`,closed-loop 并发;trace 回放用 LongBenchV2(GLM-5.1, 100K prompt, 1799 步, 78 稀疏层)。

**端到端**:
- DeepSeek-V4-Flash(2×B200, 32K/8K): 并发 64 时 600→1257 tok/s(2.1×),decode-only 1511→4308(2.9×);TTFT 并发 64 时 829s→171s;重叠区间 TPOT 相当(15.9 vs 16.0ms @并发16)。
- 长上下文吞吐扫描: Qwen+Quest GH200 32K 3.6×、200K **4.7×**(111→520 tok/s);GLM+DSA H200 32K 3.1×、160K 2.9×。4K 短上下文基本无差别。
- GLM-5.2 prefetch 研究(8×H200, 32K/8K, k=2048, B=4096): baseline 饱和后 HiSparse 各变体一直扩展到 256 并发;exact prefetch 在同并发下 **TPOT −13~15%、吞吐 +14~17%**,峰值 618→1727 tok/s(**2.8×**);**no-IO oracle 上界 2034,exact prefetch 达到 85%**(无 prefetch 74%);decode-only oracle 4671、prefetch 3410(73%)。staging 基本免费:低并发 TTFT 10.7s vs 10.7–10.8s。

**Cache 策略消融(LongBenchV2 trace)**:
- 只驻留当前 top-k(B=k)平均 **miss 30%**/步(选择集漂移);
- B=4096(=2k)下 **LRU 13.4%** vs FIFO 17.2% vs random 16.1%,走势贴合离线 **Bélady 最优 8.2%** → recency 是未来 sparse 选择的好的在线代理;
- B=8192 → LRU 6.7%(再减半),说明容量翻倍直接换 miss 减半。

**Resolve kernel 分解**: probe & scan 部分与平台几乎无关(H200 vs GH200 差 ~5%),IO 部分随 batch size 快速增长、在高 batch 下主导 PCIe;sparse attention kernel 本身 ~60µs/层(H200, per-GPU batch 8),未隐藏的 100–200µs resolve 会翻倍层关键路径 → miss 必须既少(§cache)又藏(§prefetch)。

**GH200 带宽敏感性**: B=2k、batch 16 时 GLM-5.1 的 IO 相 112→29µs/call → 快链路下大 cache 收益递减、最优点移向 B=2k。**HiSparse 把容量瓶颈变成了可调的 延迟/带宽 问题**,B 是部署时 profiling 定的静态参数(B=2k 是稳健默认)。

**局限(作者自述)**: 非容量受限场景(短上下文/低并发)只有 TPOT 开销(同步 ~7–8ms/token,带 prefetch ~3ms)没有收益,可直接禁用;真正瓶颈变成 **host 层容量**——GB200/GB300 上 Grace LPDDR ~480GB 与 HBM 相当甚至更小,容量倍数缩水(NVMe/网络层可补容量但延迟更高);anchor 层自身选择无法预知,其 miss(21/78 层 ≈ 27% IO)结构上只能同步——扣除这个地板,prefetch 藏掉了可作用 IO 的 2/3–4/5。

## 源码与本工作区的关联

- **已在本地 clone 里**: `~/source_code/sglang/` 已含完整实现——`python/sglang/srt/managers/hisparse_coordinator.py`(请求生命周期,~1000 行)、`python/sglang/kernels/jit/csrc/kvcacheio/hisparse.cuh` + `hisparse_spec.cuh`(融合 Resolve kernel,token-level 与压缩 KV 两种布局,可记录 miss plan)、`python/sglang/kernels/ops/kvcache/hisparse.py`、`mem_cache/allocator/hisparse.py`、`mem_cache/hisparse_memory_pool.py`、`mem_cache/pool_host/hisparse.py`(paged host pool)、`arg_groups/hisparse_hook.py`(配置校验)。文档 `docs/docs/advanced_features/hisparse_guide.mdx`。config: `--hisparse-config` JSON(`top_k`、`device_buffer_size`=B、`host_to_device_ratio`、swap-in block size)。
- **工程要点(附录)**: staging/write-through/prefetch 各跑独立 CUDA stream、event 排序;Resolve 与 prefetch fork 都 capture 进 SGLang 稳态 decode CUDA graph(要求所有元数据更新与 IO 发起可重放、无 host 分支);scheduler 等 staging ack 才准入,retract/pause 路径同步释放 HiSparse 状态;模型 runner 对声明了 selection-sharing group 的模型自动开精确 prefetch,PP/投机解码下降级为同步 swap-in;有 HIP(AMD)变体;disaggregated 模式下 prefill 实例通过现有 transfer backend 直接写 decode host 的 DRAM pool。**复用 HiCache 的 host-tier 基础设施(mixin),但管理对象不同**:HiCache 是跨请求 prefix 复用,HiSparse 是请求内 decode working set,两者在部署中组成 prefill–decode 对偶。
- **与 HySparse(`summary_hysparse.md`,2602.03560)对照**: 两篇都在吃"跨层共享"的红利但方向相反——HySparse 是**架构改造**(full attention 层输出 block 分数给后续 sparse 层用,训练时固化,顺带跨层共享 KV 本体),HiSparse 是**serving 系统**(不改模型,利用模型已声明的 IndexShare 分组做精确 prefetch)。GLM-5.2 IndexShare(1 indexer 带 4 层)是让 HiSparse prefetch 成为"精确而非投机"的关键模型侧条件,HySparse 的 oracle-full-attention 选择信号则从另一个角度缓解 DSA indexer 是 proxy 的问题。
- **与选择信号三部曲(`discussion_kv_selection_signals.md`)的关系**: HiSparse 是把"**选择信号本身是已知的、精确的、提前的**"这一性质变现的系统化:CacheBlend/DA/RA 讨论的是"该看谁"的信号从哪来,HiSparse 假设信号已由 indexer 给出,进一步问"**知道了每步要读谁,KV 该放哪**"。它的 LRU 命中局部性(步间重叠)与 negative-result 投机 prefetch(层间相关已被 LRU 捕获)共同说明:隐式跨层复用有限,显式共享(IndexShare)才有决定性价值。
- **与 Mooncake(`~/source_code/mooncake/`)的关系**: Mooncake 的分离式 KV 池 + 分层存储思想同构,但 Mooncake 面向 dense attention 的 prefill/decode 分离与跨实例前缀复用;HiSparse 面向 sparse attention 的**单实例请求内** decode 工作集管理。两者可叠加(HiCache host tier 与 Mooncake 传输路径互补)。
- **与 ESS(concurrent, simulation-only, 专绑 DeepSeek-V3.2 latent cache)和 ECHO(concurrent, NSA 专用、以预测 prefetch 为中心)的区别**: HiSparse indexer-agnostic 覆盖训练/免训练三族选择器,先靠 LRU 把 IO 负载降下来(而不是只想着重叠),融合 kernel 进 CUDA graph,且只对共享选择的模型做精确 prefetch。
