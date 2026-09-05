# DeepGEMM Mega MoE（MegaMoE）Megakernel 讨论笔记

- 来源：DeepGEMM main 分支源码阅读（`deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`、`comm/barrier.cuh`、`layout/sym_buffer.cuh`、`scheduler/mega_moe.cuh`）
- 背景：DeepSeek 的 EP MoE megakernel，SGLang 在 DeepSeek-V4 day-0 集成（[LMSYS blog](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/)）；运行要求 symmetric memory + 多进程 launch；`kNumMaxRanks = 72`（恰好一个 NVL72 一致性域）

## 一句话

把整个 EP MoE 层（dispatch → grouped GEMM → combine）融进一个 persistent kernel：通信被降级为普通 load/store/atomic/TMA 指令（pull 模型），同步被拆成元数据级全局 barrier + per-block flag 握手，这一切的前提是 NVLink 硬件 cache 一致性和 tcgen05 带来的 warp 资源解放。

## 为什么依赖 NVLink cache 一致性

通信模型：所有 rank 的 buffer 在 symmetric memory 堆里同地址映射（`SymBuffer::map` 只加偏移），SM 在 kernel 内直接跨 GPU 访存：

- dispatch：普通 `st.global` 把路由索引写进**远端** workspace（`sm100_fp8_fp4_mega_moe.cuh:376`）；`atom.sys.global.add` 在远端计数器做原子累加（`:395`）
- pull：TMA（`tma_load_1d`）直接从远端输入 buffer 拉 token 进本地 smem（`:548`），SF/权重普通 `ld` 读远端（`:574`/`:581`）
- combine：epilogue 普通 `st` 把结果写回远端 combine buffer（`:1299`）
- 同步：`red.release.sys` 写远端 flag + `ld.acquire.sys` 轮询（`comm/barrier.cuh` 的 `nvlink_barrier`，约 4µs）

依赖一致性的三个硬原因：

1. **release/acquire 只在一致性 fabric 上成立**：PTX `.sys` scope 的 morally-strong 语义要求互连硬件保证"观察到 flag ⇒ 数据已可见"。NVLink 硬件处理缓存可见性，软件零成本。
2. **大量远端读是普通 load/TMA**：没有一致性就得全部 uncached/volatile，或做 GPU 根本没暴露的 peer memory cache 管理。
3. **peer 显存原子操作只有 NVLink 支持**（CUDA 明确规定），而 dispatch 的计数器协议依赖它。

对比：PCIe P2P / RDMA 无此保证，只能退回 NVSHMEM 式显式 fence/quiet + 独立通信流水，无法这种粒度融合。所以 MegaMoE 限定单 NVLink 域，跨节点仍需传统 dispatch。

## 与 all2all 的区别

| | all2all（NCCL/DeepEP） | MegaMoE |
|---|---|---|
| 数据流 | push：发送方打包 staging → 批量推送 → 接收方解包，数据搬两次 | pull：发送方只写元信息，消费端 TMA 按需从远端拉，写回同理，零 staging |
| 通信载体 | 独立 kernel / collective | 普通访存指令 |
| 同步粒度 | collective 级齐步走 | 元数据全局 barrier + per-block ring 计数器 |
| 与计算 | 串行，kernel 边界气泡 | 同一调度域内 per-block 重叠 |
| 互连 | 任意（IB/PCIe 均可） | 必须 NVLink 一致性域 |

## 同步并没有消失，而是被拆碎（关键澄清）

一层内 3 个 `nvlink_barrier`（pull 前 / combine reduce 前 / workspace 清理后），**但全局 barrier 只等元数据（KB 级），不等 payload**。

payload 的同步降级为 per-block 生产者-消费者握手：pull warp 靠 TMA mbarrier transaction count 显式确认数据落进 smem，然后才 `red.release.gpu` 累加 `l1_full_count`（`:593`）；计算 warp 只轮询自己要算的那个 block 的计数（`:530`/`:694`）。

**语义澄清**：release/acquire 不是"不等数据写入确认"，而是"flag 可见性蕴含数据可见性"——数据仍然必须先到达（release 序保证 flag 不会跑在数据前面），省掉的是独立的确认机制/往返。数据依赖一个没少，省的全是协议和调度开销。

气泡小的三个真实原因：① 等待粒度小（block 而非整个 payload）；② 握手链路短（单向 flag + 硬件一致性，无往返，barrier ~4µs）；③ 等待与计算重叠（不同 warp 角色并行）。层间全局语义与 all2all 完全相同：combine 写回 → barrier → reduce → 才进下一层。

## mma → wgmma → tcgen05：megakernel 的硬件基础

先分清被"占用"的三种资源：

1. **tensor core 执行单元**（SM 共享的硬件 pipe）——做数学就占着，这是目标不是成本
2. **warp 的指令发射带宽**（issue slot）——喂养 tensor core 的编排成本
3. **SM 静态资源**（warp slot / 寄存器 / smem）

| | mma.sync (Ampere) | wgmma (Hopper) | tcgen05 (Blackwell) |
|---|---|---|---|
| 发起 | 每 warp 全员 | warpgroup（4 warp） | **单线程**（`elect_one`），可 `cta_group::2` 一条指令驱动 2 个 SM 的 tensor core |
| 执行 | 同步 | 异步（commit/wait_group） | 异步（commit + mbarrier） |
| 累加器 | 寄存器 | 寄存器（128 线程的寄存器被 C 矩阵锁死整个 mainloop） | **TMEM**，不占寄存器 |
| 操作数 | 寄存器 + ldmatrix | smem descriptor | smem descriptor + TMEM |

- mma.sync 的真实问题不是"别的 warp 没机会执行"（4 个 scheduler 照常调度），而是**保持 tensor core 满载需要抵押整个 warpgroup 的发射带宽 + 寄存器**，SM 上没剩资源给通信/调度。
- wgmma：吞吐翻倍（SM 级共享 tensor core）逼出 warpgroup 粒度；event 机制让 warp 发射完能走，但寄存器仍被累加器押着。
- tcgen05：发射成本降到单线程，累加器进 TMEM。注意两个术语澄清：**是"一个线程"不是"一个 cuda core"**（发射走 warp scheduler 常规路径）；**累加从来是 tensor core 算的**，wgmma 时代的问题是累加**结果存放在寄存器文件**。CUDA core 并非"不做任何计算"——epilogue（缩放/激活/类型转换/stmatrix）、NVLink 写回、调度全是 CUDA core，只是退出了 GEMM 主数据通路。

对 MegaMoE 的直接作用：

1. warp 角色分化的前提：tensor core 的"喂养成本"从整个 warpgroup 降到单线程，省下的 issue slot 才能养 dispatch/pull/scheduler/epilogue warp（源码 `:315-322` 用 `warpgroup_reg_dealloc/alloc` 在角色间重切寄存器预算，这只有在累加器不占寄存器后才可行）
2. mbarrier 统一了 TMA 搬运、MMA 完成、跨 GPU flag 的同步原语 → 计算变成数据流图的普通节点，ring buffer 流水线得以串起来
3. CTA-pair cluster MMA（B 操作数经 cluster smem 共享，省一半 smem 流量）→ 调度器按 CTA-pair 组织任务（`kNumCTAsPerCluster=2`）
4. 异步发射 + persistent kernel + on-device scheduler（原子任务计数器）→ MoE 每 expert 变长的 grouped GEMM 几乎不让 tensor core 空转

代价：异步化把正确性负担全推给软件——TMA/MMA/TMEM/远端写之间的 ordering 全靠手写 fence 和 mbarrier 相位管理（`tma_store_fence`、`fence_view_async_tmem_load`、`tcgen05.commit` 等）。

## 总结链条

NVLink 一致性（通信 = 普通访存）+ tcgen05（计算发起/完成近乎免费且异步）+ symmetric memory（同地址映射）三者叠加，才使"一个 kernel 内计算、通信、调度同处一个调度域、按 block 粒度重叠"成立。生产部署的组合：节点内 MegaMoE pull，节点间仍走 RDMA all2all。
