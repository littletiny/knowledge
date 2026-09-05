# DeepGEMM：DeepSeek 的统一 Tensor Core Kernel 库（源码阅读笔记）

- 来源：无独立论文（仅 GitHub 发布，citation 为 @misc；作者 Chenggang Zhao、Zhean Xu、Liang Zhao、Jiashi Li 等）。本地仓库 `~/source_code/DeepGEMM`，commit 559d79f（Public release 26/07）
- 姊妹篇：Mega MoE megakernel 的深入分析见 `summary_megamoe_megakernel.md`，本文档覆盖库的其余部分

## 一句话

把现代 LLM 的关键计算原语（FP8/FP4/BF16 GEMM、EP 融合 Mega MoE、lightning indexer 的 MQA scoring、HyperConnection）收进一个轻量 CUDA 代码库：借鉴 CUTLASS/CuTe 概念但不依赖其模板体系，全部 kernel 经运行时 JIT 编译（安装期零 CUDA 编译），启发式结果全部固化为编译期常量。

## 项目定位与演进

- 支持 SM90（Hopper）与 SM100（Blackwell）；CUDA 12.3+（SM90，强烈推荐 12.9+）/ 12.9+（SM100）。
- 2025.02 首版（开源周）：SM90 FP8 GEMM + 细粒度缩放，H800 上达 1550 TFLOPS。
- 2025.07：SM100 支持 + 全面重构为低 CPU 开销的 C++ JIT 模块；**NVRTC 与 SASS 后优化全部禁用**——NVCC 12.9 已自动做 FFMA interleaving（README.md:17-19）。
- 2025.09：DeepSeek-V3.2 lightning indexer 的 MQA logits kernel。
- 2026.04：Mega MoE、FP8xFP4 GEMM、FP4 indexer、PDL。

## SM90 FP8 GEMM 的核心技术

### 两级累加（解决 Hopper FP8 累加精度不足）
- Hopper tensor core 的 FP8 累加精度有限，DeepGEMM 的办法：**WGMMA 累加器每个 BLOCK_K=128 的 k-block 清零重算**（该 block 第一条 WGMMA 传 ScaleOut::Zero，`mma/sm90.cuh:22-28`、`impls/sm90_fp8_gemm_1d1d.cuh:297-301`），`warpgroup_wait<0>` 后立即由 CUDA core 把 `scale_a × scale_b × accum` 累加进 FP32 的 `final_accum`（`sm90_fp8_gemm_1d1d.cuh:306-320`）。
- BLOCK_K==128 是硬约束：scale 粒度本来就是 per-128-channel，提升点与 scale 应用点天然重合。tensor core 内最多只累加 128 个 K 元素，误差被限制在一个 scaling block 内。

### Warp specialization
- 128 线程 TMA warpgroup（实际 1 个 elected thread 发 TMA）+ 128/256 线程 math warpgroups；全程 persistent + on-device scheduler 供 block。
- 每 stage 一对 full/empty mbarrier（ClusterTransactionBarrier），phase 翻转推进；TMA 用 `arrive_and_expect_tx` 带字节数。stage 数由 smem 容量（232448B）反推，上限 16。
- 寄存器重配置：TMA warps dealloc 到 24/40，math warps alloc 到 232/240（`sm90_fp8_gemm_1d1d.cuh:154-172`）——mega kernel 里角色寄存器重切的做法在普通 GEMM 里就有了。

### Fine-grained scaling 的两代对比
- SM90：scale 在 CUDA core 提升时乘入，每个 128-K-block 一次。1D1D（SFA/SFB 均 1D FP32）输出 FP32 并用 TMA reduce-add 实现 D=C+A@B 的原子加；1D2D（SFB 为 128×128 块 scale）输出 BF16，SFB 由 math warps 提前搬进 smem 与上一个 block 的 TMA store 重叠。
- SM100：**累加与 scale 全部硬化**——UMMA block-scaled 指令直接从 TMEM 读 UE8M0 SF 在 tensor core 内乘入，FP32 累加在 TMEM，没有 SM90 的 CUDA-core 两级提升（`sm100_fp8_fp4_gemm_1d1d.cuh:374-392`）。SF 路径：TMA → smem → warp 内 4×32 转置 → `UTCCP_4x32dp128bit` 拷进 TMEM → UMMA 经 instruction descriptor 的 `a_sf_id_/b_sf_id_` 消费。
- UE8M0 packed 格式：主机侧把 4 个只取指数的 FP32 scale 沿 K 打包进 1 个 uint32（`impls/smxx_layout.cuh:130-142`）。
- SM100 只有 1D1D 内核，因为硬件 block scaling 原生支持 gran_k ∈ {32(MX), 128(DeepSeek)}（`sm100_fp8_fp4_gemm_1d1d.cuh:65-68`）。

### SM100 其他要点
- cluster=1 用 `SM100_MMA_MXF8F6F4_SS`，cluster=2 用 `2x1SM_SS` 变体（一条指令驱动 2 SM 的 tensor core）；只有 leader CTA 的 warp 1 发 MMA。
- 线程配置固定 256：warp0 TMA / warp1 MMA / warp2 UTCCP / warp3 分配+空闲 + 128 epilogue（`heuristics/sm100.hpp:219-226`）；TMEM 512 列是硬约束。

## Grouped GEMM（MoE 专用）

- **只沿 M 轴分组**（N/K 固定，因为各 expert 同构），两种 layout：
  - **Contiguous**（训练前向/推理 prefill）：各 expert token 拼接成一个张量，每段 M 对齐（默认 128；SM100 理论最小 224，可按 expected_m 缩到 32 步进）。`grouped_layout` 是每行的 expert index，scheduler 用 block 首行 index 给 B 加偏移；相邻 CTA 属于不同 expert 时动态禁用对 B 的 TMA multicast。
  - **Masked**（decode + CUDA graph，CPU 不知各 expert token 数）：形状 [G, M, K]，`grouped_layout` = 每组有效 token 数；mask 在 warpgroup 粒度生效（`m_offset < masked_m` 才算）。SM100 swap-AB 下用动态 UMMA_N = 对齐后有效 M。DeepEP low-latency kernel 的输出可直接作输入。
- **K-grouped**（MoE weight 反传）：沿 K 拼接，group 切换时在 smem/gmem 原地改写 TMA descriptor 的地址和内维 stride（`sm90_fp8_gemm_1d1d.cuh:191-215`）。

## JIT 架构

- 每个 kernel 一个 `LaunchRuntime` 子类：`generate_impl` 用 fmt 拼出 `.cu`，`__instantiate_kernel` 里对模板取地址完成显式实例化；**所有 heuristic 结果（block size、stages、swizzle、cluster、num_sms、dtype）都变成编译期常量**。
- 缓存 key = hash(name + compiler signature + flags + code)，目录 `~/.deep_gemm/cache/`；生成代码头部注入所有 include 头文件的递归 hash，头文件改动自动失效缓存。编译到 tmp 后整体 rename 原子发布 + fsync（兼容分布式文件系统）。
- 加载优先 `cuLibraryEnumerateKernels`，fallback cuobjdump 解析符号表；launch 走 `cudaLaunchKernelExC` 带 cluster/PDL attribute。
- Heuristics：SM90 用 L1/L2 cycle 成本模型 ÷ wave 效率选优；SM100 按"单 wave → 大 cluster → 少 wave → 末 wave 利用率 → 更小 block"优先级。

## MQA logits（lightning indexer，DeepSeek-V3.2）

- 语义：`out[i,j] = Σ_h ReLU(q[i,h,:]·kv[j,:]) × w[i,h] × kv_scale[j]`。
- 技巧：把 `[BLOCK_Q×heads, head_dim] @ [BLOCK_KV, head_dim]ᵀ` 当成 GEMM 做（KV 作 M 侧），WGMMA/UMMA 之后在 CUDA core 做 ReLU×weights→归约→×kv scale→直接 st.global。ReLU 用 `(x+|x|)/2` 的 float2 FMA 序列实现。
- MXFP4/MXFP8 路径把 scale 折进 block-scaled UMMA，无额外乘法。Paged（decode）版由单独 metadata kernel 预计算调度信息，TMA 按页发 3D copy。

## 其他工程细节

- **PDL**：所有内核开头 `cudaGridDependencySynchronize()`，launch 侧按需加 ProgrammaticStreamSerialization；默认关闭（`set_pdl`）。
- **set_tc_util**：近似 tensor core 利用率上限（0-100）。只有 SM100 BF16 GEMM 消费：每个 k-block UMMA commit 后 busy-wait `(100-util)/util × UMMA_cycles` 个时钟，注释说是"让 tensor core 休息以降低降频概率"（`impls/sm100_bf16_gemm.cuh:370-388`）。
- **BF16 vs FP8xFP4 Mega MoE**：骨架一致，BF16 版无 SF/UTCCP、UMMA_BLOCK_K=64；FP8xFP4 版 UMMA_BLOCK_K=128、需 2-CTA UTCCP 送 UE8M0 SF 上 TMEM，routed/shared expert 用不同 instruction descriptor，L1 输出经 SwiGLU 后量化回 FP8。
- 用户侧责任：输入转置、FP8 cast 等前置操作需自行融合进上游 kernel，库只优化 GEMM 本体（README 明确声明）。

## 与 MegaMoE 讨论的联系

普通 GEMM 内核已经具备 MegaMoE 的全部基础件：persistent + on-device scheduler、mbarrier stage 流水线、warp 角色寄存器重切、TMEM 累加、CTA-pair UMMA。MegaMoE = 这套骨架 + symmetric memory + NVLink 一致性下的跨 rank flag 协议（详见 `summary_megamoe_megakernel.md`）。SM90→SM100 的"两级累加 → 硬件 block scaling"演进也印证了同一条主线：围绕 tensor core 的软件编排负担被逐代卸给硬件。
