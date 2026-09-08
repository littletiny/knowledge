# 消费级 GPU 跑前沿 MoE:SM89 的 FP8/sparse-MLA 问题与四种解法对比

- **日期**: 2026-09-08
- **相关文档**(详细总结见各自文件):
  - FreeToken(arXiv 2608.16157)→ `summary_freetoken_edge_moe.md`
  - KTransformers(SOSP 2025)→ `summary_ktransformers_sosp25.md`
- **涉及源码**(`~/source_code/`):`vllm/`、`sglang/`、`FreeToken/`、`ktransformers/`、`ds4/`、`llama.cpp/`、`h3.c/`
- 本文档记录一次完整调研:从"SM89 怎么跑 FP8 模型"出发,落到 sparse-MLA 这个真实卡点,再横向对比 FreeToken / KTransformers / ds4 / llama.cpp 四种消费级部署方案的设计取舍。

## 0. 问题起点与结论速览

**起点问题**:SM89(RTX 4090/L40,Ada)如何跑 FP8 模型?能否权重低比特存储、计算用 bf16?

**速览结论**:

1. SM89 有 FP8 tensor core,vLLM/SGLang 对 W8A8 FP8 per-channel 走原生 CUTLASS SM89 kernel(需 CUDA ≥ 12.4),"4090 无 FP8 硬件"的说法是错的;
2. 真正的卡点是 **block FP8(DeepSeek 式 128×128)和 sparse MLA**:DeepGEMM/FlashMLA sparse 都要 SM90+,SM89 在上游框架里掉到 Triton/Marlin 兜底或直接没有路径;
3. "权重 W4/W8 存储 + bf16 计算"是成熟方案(GPTQ/AWQ-Marlin、FP8 Marlin W8A16),纯配置可用;
4. `--kv-cache-dtype` 不放宽 backend 的 capability 门槛,救不了 SM89;SGLang 可用 `--json-model-override-args '{"index_topk":null}'` 关稀疏走 dense Triton MLA(近似,行为偏离);
5. 社区已有可用解:yhfgyyf 的 vLLM fork(定制 FlashInfer SM89 sparse kernel,8×4090 48GB 跑通 DS-V4-Flash/GLM-5.3-Flash)、FreeToken(自研纯 Triton sparse MLA)、ds4(SSD streaming)。

## 1. vLLM/SGLang 在 SM89 上的真实现状

### 1.1 FP8 GEMM:原生支持

- vLLM:`cutlass_scaled_mm_supports_fp8` 对 cc ≥ 89 返回 true(`csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_entry.cu:145-159`);W8A8 FP8 走原生 CUTLASS SM89 kernel。FP8 Marlin(W8A16,fp8 存储 + fp16 计算,cc ≥ 7.5)只兜 SM < 89。
- SGLang:`cutlass_fp8_supported()` 对 SM89 + CUDA ≥ 12.4 返回 True(`python/sglang/srt/layers/quantization/fp8_utils.py:271-281`),FP8 Marlin 自动回退范围是 `80 <= sm < 89`(`can_auto_enable_marlin_fp8`)。
- **block FP8(128×128 block scale)是例外**:DeepGEMM 要求 SM90+,SM89 上 vLLM 落 Marlin block 路径(仅 128×128)/Triton block kernel,SGLang 落 Triton(`w8a8_block_fp8_matmul_triton`)——功能可用、无 tensor core 加速。

### 1.2 Sparse MLA:真正的墙

DeepSeek-V4-Flash / GLM-5.3-Flash 的 sparse MLA(indexer + top-k)在上游没有任何 SM89 路径:

- vLLM:全部 sparse MLA backend(FlashMLA Sparse / FlashInfer sparse 各变体)要求 SM90/100/120;唯一支持 SM89 的 TRITON_MLA 是 dense-only,被 `use_sparse != cls.is_sparse()` 校验拒绝(vllm-project/vllm#54059 的报错现场)。DS-V4 甚至**拒绝 bf16 KV**(`models/deepseek_v4/attention.py:90-125`,fp8_ds_mla layout 下传 bfloat16 直接 raise)。
- SGLang:DSA 子后端 flashmla_sparse 是 SM90-only,fa3 对 MLA 要求 Hopper;GLM-5.2/5.3 可 `--json-model-override-args '{"index_topk":null}'` 关稀疏走 dense Triton MLA + bf16 KV(官方文档支持的近似),DS-V4 的 dsv4 backend 硬依赖 FlashMLA,SM89 无路。

**kv-cache-dtype 只改存储格式、不放宽 capability**——dtype 不是开关能解决的问题,缺的是 kernel 本身。

## 2. FreeToken:动态缓存 + 带宽自适应(详见 summary_freetoken_edge_moe.md)

- **sparse MLA 解法**:自研纯 Triton kernel(`kernel/triton/dsv4/sparse_attn.py`、`glm_dsa_sparse.py`),bf16 KV 池 + fp32 累加,整条路径零 SM90 依赖(所以覆盖到 SM86/3090);SM86 的 FP8 缺口用 `e4m3_compat.py` 位级软件模拟补。
- **量化**:weight-only 分层——SM89 上 block-FP8 prefill 用原生 FP8 tensor core,decode 一律 W8A16;FP4(NVFP4/MXFP4/DS-FP4)从不用 FP4 tensor core,kernel 内 dequant 到 bf16 算。
- **架构核心**:expert 全量在 host pinned 内存,GPU 上是**图兼容的运行时 LRU slot cache**(簿记全为设备驻留固定形状张量,miss 决策/切分/victim 选择在 CUDA graph 内完成);q\* 策略按实测 PCIe/CPU 重叠带宽比把 miss 切成"同时完成"的两份(`q* ≈ m·B_pcie/B_host`);prefill 全层双缓冲把搬运打成纯传输受限。
- **缓存管理**(设计密度最高的部分):ShadowRadix 分层(共享 page_table 虚拟坐标 + 各 tier 算术派生)、每 attention 族一个 KV 池(DSV4 的 window ring+压缩池+indexer+fp32 状态环,GDN 的每请求状态 slot 池)、radix 前缀树的 SWA/GDN 双货币变体、**语义锚点 checkpoint**(tool-call opener 处保复用点)、运行时弹性重建(破坏前预检、teardown graph、原地 resize、回滚)。
- **全显存模式**:`--moe-backend fused` 存在但只支持 bf16/fp8_block expert(DS-V4 的 FP4、GLM 的 NVFP4 走不通,GLM loader 直接 assert 拒绝);实际是 offload + `--moe-cache-rate 1.0` 让 LRU 全覆盖。TP 支持(`--tp-size` + 每 rank 一个 `--gpu`)。

## 3. KTransformers:静态分工 + 强 CPU(详见 summary_ktransformers_sosp25.md)

- 论文(SOSP'25)对应现已进 `archive/` 的注入式框架;现役主线是 **kt-kernel 库 + SGLang fork**(调度/attention/KV 全归 SGLang,KT 只做 MoE expert 的 CPU-GPU 分流)。
- 与 FreeToken 的根本分歧:**静态 expert 放置**(离线 profiling 的热 expert 掩码 + shared expert 驻 GPU)vs 动态 LRU;**Expert Deferral**(低分 expert 推迟一层合并,精度换重叠,LiveBench 掉 0.5%)vs 精确执行红线。
- CPU kernel 投入深得多:AMX tile 感知布局(单路 21.3 TFLOPS,oneDNN 只到峰值 7%)、ARI 自适应 AMX/AVX-512 切换、NUMA 感知 TP。**主场是双路至强(440GB/s DRAM)+ 单卡;消费级双通道(50-90GB/s)不是它的目标**——FreeToken 评测把 CPU 压到 6-8 线程后 KT 明显吃亏,两边互评都要打折。
- 相同点:都用 `cudaLaunchHostFunc` 把 host 回调嵌进 CUDA graph(独立发现的同一招)。

## 4. 讨论沉淀:CPU 到底 offload 什么?

- **CPU 上只跑 routed expert 的完整 FFN**(gate/up → 激活 → down),attention/router/shared expert/dense 层全在 GPU。两家一致。
- **动机排序:显存容量 > DRAM 带宽利用 > FLOPs**。decode 是 GEMV(计算强度 ~1 FLOP/byte),CPU 的价值是"内存带宽提供者",本质是 near-memory computing——省的是 PCIe 搬运,不是算力。只有 prefill(高 ARI)且 CPU 侧带宽×算力乘积够强(KT 的 AMX 场景)时,CPU 才作为真实算力参与 prefill;FreeToken 的 prefill 则全量流给 GPU。
- **容量下界**:交互式速度要求 模型权重 ≤ DRAM + VRAM(NVMe ~7GB/s 比 DRAM 慢一个数量级以上);速度由热点驻留率决定。ds4 把下界推到 SSD(见下),多机池化(Mooncake/NIXL 思路)是另一个方向。

## 5. ds4(antirez):SSD streaming + 模型专精单文件引擎

- 定位与 FreeToken/KT 不同:**第一层就承认模型 > 内存**——非 routed 权重常驻内存,routed expert 放内存 cache,miss 从 GGUF 文件(SSD)流式加载;热 expert hotlist 预载。约束放宽为"非 routed 部分 + KV + expert cache < 内存,总量 < SSD 容量"。
- 敢这么做的前提:**非对称 2-bit 量化**(routed expert IQ2_XXS/Q2_K,其余不动)把每 token 活跃字节压到 Mac SSD(~5-7GB/s)能容忍的范围。
- **kernel 策略** = 自研(Metal/ROCm 全部、CUDA 的 attention/MoE 调度)+ 逐行 vendored llama.cpp 的量化 GEMM(`cuda/mmq/`,~1.1 万行 + 600 行 shim,pin commit + re-sync 手册)+ cuBLAS/hipBLASLt dense GEMM。不链接 ggml,保持自包含。
- **UMA 推論**:Mac/Strix Halo 没有 VRAM/PCIe 边界,CPU compute offload 的存在理由消失——Metal 直接读统一内存即可,ds4 的 streaming miss 也是加载后喂给 Metal 而非 CPU 算。CPU 后端的价值 = "存在独立带宽域"或"GPU 不存在"。

## 6. llama.cpp 上游现状(2026-09 核对,修正旧印象)

- **GDN/KDA(FLA 系)早有**(Qwen3-Next day-zero、Kimi Linear);dense MLA 早有。
- **DSA sparse attention 已上游化**:05-29 `deepseek32.cpp`(通用 DSA,#23346)→ 06-29 `deepseek4.cpp`(V4 完整 indexer/压缩器,#24162)→ 09-02 CUDA sparse-FA(#27970)→ **09-03 Metal sparse FA(#28098)**。
- 实现手法:不写新 kernel,给 flash-attn 加 `ggml_flash_attn_ext_set_n_kv_max` hint——**mask 的有限值条目即稀疏 K/V 集合**,mask 是事实源(所有后端天然正确),n_kv_max 只是 CUDA/Metal 的加速索引表尺寸上界。Metal 版覆盖 dk/dv 576/512(MLA 形状)和 Q4_0/Q8_0 量化 KV,n_kv_max ≤ 4096。CPU 后端仍是 dense-mask 路径。
- **h3.c**(antirez 的 MiniMax-H3 视频生成)算子对照:≈100% 被 ggml 覆盖(含 Conv3D direct、ConvTranspose1d),硬缺口无;软缺口是 Metal PAD 不支持左 pad、Snake 激活需组合、以及失去手写融合 kernel 的性能。**llama.cpp 的护城河是算子/量化基座,专精引擎赢在融合、内存策略和垂直整合。**

## 7. 方案速查表

| | FreeToken | KTransformers | ds4 | llama.cpp(上游) |
|---|---|---|---|---|
| 形态 | 独立引擎 | kernel 库 + SGLang fork | 模型专精单文件 C | 通用引擎 |
| expert 放置 | 动态 LRU + q\* | 静态掩码 + deferral | 内存 cache + SSD streaming | 静态分层/offload |
| 近似 | 无(精确红线) | deferred expert(~0.5%) | 无 | 无 |
| 容量下界 | DRAM+VRAM | DRAM+VRAM | SSD(非 routed 入内存即可) | DRAM+VRAM |
| 主目标硬件 | RTX 30/40/50 + 普通桌面 CPU | 双路至强 AMX + 单卡 | Apple Silicon(Metal 优先) | 全平台 |
| sparse MLA | 自研 Triton(全 SM) | 委托 SGLang(SM90+,SM89 Triton 兜底) | 自研 C/Metal | 上游已有(CUDA/Metal 真稀疏) |

## 8. 未验证/待跟进

- yhfgyyf vLLM fork 的 SM89 sparse kernel 是否会上游化(对应 vllm#54059 官方征求贡献);
- llama.cpp CPU 后端的 sparse-FA(目前 dense-mask 路径,长上下文 GLM-DSA 会贵);
- FreeToken 论文与 KT 论文互评的硬件前提差异(CPU 线程封顶)在真实桌面 AMX 缺失场景下的复测;
- ds4 的 SSD streaming 在 agentic 长会话(高 prefill 频率)下的尾部延迟表现。
