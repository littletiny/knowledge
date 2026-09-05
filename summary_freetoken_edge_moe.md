# FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution

- arXiv: 2608.16157（2026-08），UC Berkeley / MIT / UT Austin 等（Song Han、Ion Stoica、Matei Zaharia、Kurt Keutzer 等在作者列表）
- 代码: https://github.com/FlashML-org/FreeToken （本地已 clone 至 `FreeToken/`，论文与代码已交叉验证）

## 一句话

把个人电脑（GPU + CPU + DRAM + PCIe）当作一个统一的弹性推理平台：专家池常驻主机内存，GPU 上维护全层共享的 LRU expert cache，prefill 用全层双缓冲把权重搬运藏在计算后面，decode 用闭式 q* 策略把 cache miss 按**实测带宽比**在"PCIe 取回 GPU"和"CPU 就地计算"之间切分，整条异构路径捕获进 CUDA Graph。

## 问题定义（论文的三个挑战）

1. **Prefill 破坏 MoE 的工作集稀疏性**：长 prompt 的 token 并集几乎激活每层全部 expert，FP4 的 DeepSeek-V4-Flash 每次 prefill 要搬 ~140GB expert 权重（PCIe 5.0 x16 约 +2s，PCIe 4.0 约 +5s，笔记本 x8 超 10s）。Agent 场景每轮工具调用都触发 prefill，雪上加霜。
2. **Decode 的 miss 服务没有原则性策略**：静态放置（llama.cpp 按层、KTransformers 钉热 expert）追不上逐 token 变化的路由；纯预测/预取只能降 miss 率，不能决定"不可避免的 miss 如何在 PCIe 传输 vs CPU 计算之间分配"。消费级 CPU 双通道 DDR4/DDR5 只有 50-90GB/s，单独扛不动 decode。
3. **边缘资源异构且不专用**：VRAM 会被浏览器/游戏动态挤占，KV cache 需求随会话增长，引擎频繁启动（140GB 从 NVMe 读入就要 ~20s）。静态放置策略无法跨机器、跨阶段、跨运行条件工作。

## 核心设计

### 两级 expert 存储层级
- **CPU Expert Pool**（主机内存，source of truth）+ **GPU 端全层共享的弹性 LRU slot cache**：slot 以逻辑 (layer, expert) 为单位，一个 slot 装一个 layer-expert 对的全部 tensor。
- 非 expert 权重（attention、dense、shared expert、router、embedding/lm_head）常驻 GPU。

### Prefill：全层双缓冲 + 语义锚点
- 不按 demand 取 expert，而是从 slot 池借两个全层 buffer：GPU 算第 l 层时，独立 copy stream 把第 l+1层**全部** expert 流进来（不需要等路由结果）。与 decode cache 共享 slot 池，prefill 结束幸存的 expert 直接给 decode 热启动。slot 池不够两个全层时退回按需加载。
- **Semantic-Aware State Cache**：hybrid attention 模型（SWA/GDN/KDA 等循环层）的前缀复用依赖循环状态 checkpoint。checkpoint 打在**语义边界**（thinking 段、工具调用、会话轮次的 special token 处）——agent 框架（OpenClaw/OpenCode/SWE-agent）编辑上下文时恰好按这些块为单位删改，锚点处的 checkpoint 存活率远高于任意位置。编辑后只重算真正新的后缀。

### Decode：q* 带宽自适应切分（核心公式）
- 每步 m 个 miss 分成 FillSet（PCIe 取回入缓存，GPU 算，留下复用）和 CpuSet（CPU 就地算，不改驻留）。
- 两条路径共享同一主机内存子系统：饱和 PCIe 后残余带宽 B_residual = max(B_host − B_pcie, 0)。平衡两条并发分支：
  - T_fill(q) ≈ qS/B_pcie，T_cpu(m−q) ≈ (m−q)S/(B_host−B_pcie)
  - **q\* ≈ m · B_pcie / B_host**
- B_host、B_pcie 是在目标机上实测的（`ft bench bw`，且是**并发竞争下**的重叠带宽）。B_host → B_pcie 时退化为纯按需缓存填充。
- **精确执行**：不改路由、不跳过/替换/近似 expert（对比 HOBBIT/SiDA/SMoE 的降精度取回或跳 expert）。
- 执行顺序：先发 CPU 分支，再跑 GPU miss 路径（缓存更新 + FillSet 批量拷贝 + HitSet∪FillSet 的 grouped GEMM），CPU worker 并发处理 CpuSet。

### 图兼容实现（工程难点）
- 所有路由相关控制（去重、驻留判定、q 值计算、victim 选择、CPU 分支提交）都是**设备驻留数据**（固定形状工作缓冲 + 设备端计数），捕获进静态 CUDA graph，replay 即执行完整异构步骤，无逐 token Python 调度。
- Victim 选择单遍 kernel 一次找出 K 个最久未用 slot，miss 路径只消费前 q 个——无论 miss 多少都只付一遍扫描。
- CPU worker 是钉物理核的持久 C++ 池，SIMD + kernel 内反量化，返回 gate 加权 partial sum。

### 弹性内存与快速启动
- CPU pool 是 source of truth，所以 GPU 缓存只影响性能不影响正确性 → 可在 scheduler safe point **运行时重建** expert cache（VRAM 预算变化时不重启引擎、不重载 host 池）。
- 启动优化：expert 从磁盘直接读入最终 host 布局，填完再 pin（先 pin 空 buffer 会白 fault 几十 GB）；无需预热，冷缓存直接服务。

## 实验要点

- 6 台机器（8GB RTX 4060 笔记本 → 单卡 RTX PRO 6000 96GB），模型 Qwen3.6-35B-A3B（BF16）、DeepSeek-V4-Flash（原生 MXFP4 expert）、GLM-5.2（753B，NVFP4，433GB checkpoint）；4 个真实 agentic 负载（AIME 推理、OpenCode/Claude Code 编程 agent、OpenClaw 邮件 agent）；baseline 为 llama.cpp/Ollama/KTransformers/MoE-Infinity，权重格式严格对齐（bit-exact）。
- RTX 5090：Qwen3.6 77-83 tok/s、DSV4-Flash 22-25 tok/s，decode 吞吐为最强 baseline 的 1.5-2.3x；agentic 负载下性能衰减 <12%（KT 在 W2 已掉 31%）。
- 尾部 TTFT：FreeToken 所有场景 <44s；baseline 至少有一处 >150s（KT 最差 946s），越过真实客户端的超时阈值（OpenClaw 120s watchdog）。
-  locality 量化（同等缓存容量回放路由 trace）：FreeToken 全局 LRU miss 率 16%/39%（Qwen3.6/DSV4），KT 的 prefill 更新放置 41%/59%，llama.cpp 静态按层 62%/89%。5090 上缓存容量分别为 expert 池的 37%/11%。
- 双缓冲 ablation：关掉后 prefill 吞吐掉 19-26%（随 prompt 变长惩罚增大）；开 overlap 时 8K chunk 1.19-1.22s ≈ 以 52.7GB/s 流一遍 64.4GB expert 池的时间（PCIe 5.0 x16 实际上限），即 prefill 变成纯传输受限。
- 跨硬件：W2 负载下领先最强 baseline 1.3x（3090/4090）~2.1x（5090 桌面）；4060 笔记本 39.3 tok/s；同 GPU 换主机（服务器多通道→桌面双通道）FreeToken 只掉 4%，llama.cpp 掉到 80%。
- GLM-5.2 单卡 RTX PRO 6000：14.9 tok/s vs llama.cpp 7.3（2x）；KT 在此机无可用路径（其 GLM-5.2 方案需 753GB-1.5TB 主机内存 > 机器 512GiB，且 CPU kernel 不认 NVFP4 布局）。
- 评估方法学细节：租用双路服务器的 CPU 被限制到 6-8 线程并钉在 GPU 所在 NUMA 节点，把主机带宽压到 56.7-77.3GB/s 以模拟真实边缘机；笔记本/桌面真机验证该仿真成立。

## 与我们本地代码阅读的对照（vLLM/SGLang/KTransformers 背景）

- 论文的 q\* 公式与代码实现一致：`moe/offload_kernels.py:344-355` 的 `_ensure_experts_hybrid_kernel` 用 Q16 定点在 kernel 内算整数切分（选使 max(两侧耗时) 最小的整数邻居）；fraction 来自 `bench_profile.py:156-191` 的重叠实测带宽对（`pcie_ov/(pcie_ov+cpu_ov)`）。
- 论文说"launches the CPU branch first"；代码里对应 shared expert GEMM 先入流再处理 routed（`models/deepseek_v4/moe.py:140-150`）+ `cudaLaunchHostFunc`/mapped-pinned flag 握手。
- 论文未覆盖、但代码里很关键的 SM89/SM86 适配：自研纯 Triton sparse MLA kernel（`kernel/triton/dsv4/sparse_attn.py`、`glm_dsa_sparse.py`）绕开 FlashMLA 的 SM90 限制；SM86 的 e4m3 位级软件模拟（`kernel/triton/e4m3_compat.py`）。这是它能覆盖 RTX 30 系列的底层原因，论文里只在 Implementation 一笔带过。
- 论文称 DSV4-Flash 的 routed expert "natively MXFP4"；代码里对应 `ds_fp4`（float4_e2m1fn_x2 + e8m0 per-32 scale，本质即 MXFP4 变体）。

## 对我们（消费卡跑大模型）的启发

1. **miss 服务策略比 miss 预测更重要**：预测类工作（MoE-Infinity/ProMoE 等）只能降 miss 率， decode 延迟仍被 PCIe 封顶；把 miss 当"可以在原地算的工作"（Fiddler 首创，KT 用 AMX 做实）并闭式分配才是正解。
2. **静态放置在 agentic 负载下会崩**：KT 在 W2 掉 31% 的主因就是放置不随上下文漂移；LRU 这种最朴素的逐 miss 策略就能反超，因为工作集随话题/工具走。
3. **CPU 的角色是"内存带宽提供者"而非算力**：decode GEMV 计算强度 ~1 FLOP/byte，CPU 的价值是 DRAM 带宽（50-90GB/s）超过 PCIe 有效带宽的部分；q\* 本质是把这个残差带宽兑换成当前 token 进度。
4. **CUDA graph 兼容是工程分水岭**：host 控制的缓存在每个 MoE 层引入同步；把控制面全部变成设备驻留数据才能保住 graph replay 的低开销。
5. **语义锚点 checkpoint** 是个可移植的想法：任何支持 hybrid attention（SWA/linear attention）的 serving 栈都可以把循环状态 checkpoint 打在 special token 边界，专门服务 agent 工具的上下文编辑模式。
6. 局限性注意：论文评测全部把 CPU 线程压到 6-8 核模拟边缘机；KT 的 AMX 路径（需要至强 AMX 芯片）不在其支持矩阵内时 KT 吃亏明显，对比解读要带上这个前提。
