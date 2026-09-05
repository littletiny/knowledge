# KTransformers: Unleashing the Full Potential of CPU/GPU Hybrid Inference for MoE Models

- SOSP 2025（首尔），清华 MADSys + Approaching.AI（趋境）等；DOI: 10.1145/3731569.3764843
- 未上 arXiv；PDF: https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf
- 代码: https://github.com/kvcache-ai/ktransformers （本地 `ktransformers/`；注意论文对应的是现 `archive/` 下的注入式框架，现役主线已重构为 kt-kernel + SGLang fork）

## 一句话

针对"routed expert 放 CPU 就地算"（Fiddler 式）混合推理的三大瓶颈——CPU 算力闲置（oneDNN 只发挥 AMX 峰值 7%）、CPU-GPU 同步开销（Fiddler 每 token 7000+ 次 kernel launch，73% GPU 时间花在 launch 上）、跨 NUMA 访问低效——给出 AMX 专用 kernel + 单 CUDA Graph 异步调度 + Expert Deferral 延迟专家计算，prefill 提速 4.62-19.74x、decode 提速 1.66-4.90x。

## 问题量化（motivation 部分的实测数字）

- Fiddler 式混合推理跑 DeepSeek-V3（671B）：prefill 仅 70 tok/s、decode 4.68 tok/s、GPU 利用率 <30%。
- Kernel launch 分析：Fiddler 每 decode token 触发 7000+ 次 CUDA kernel launch，平均 16µs/次，占 GPU 执行时间 73%；llama.cpp 压到 ~3000 次、5µs/次，仍占 21%。
- CPU 微基准：PyTorch/oneDNN 的 AMX kernel 只到 AMX 理论峰值（73.7 TFLOPS）的 7%（5.4 TFLOPS）；跨 NUMA socket 访问使双路只比单路快 16%。
- PCIe 4.0 32GB/s vs 双路 Xeon DDR5 440GB/s —— 权重搬运不如就地计算（这也是 FreeToken q* 策略面对的同一本账）。

## 核心技术

### 1. Arithmetic Intensity-Aware 混合指令 kernel（§3.2）
- **AMX tiling 感知内存布局**：加载时把 expert 权重预处理成 AMX tile（16 行 × 64 字节）兼容的子矩阵，64B 对齐，group-wise INT8/INT4 量化 scale 分离存放，消除运行时 transpose。
- **Cache 友好 AMX kernel**：权重竖切任务动态分配到线程、横切到 L2 大小的 block、tile 级 AMX 计算；输入驻留 L3、权重只从 DRAM 读一次。单路 21.3 TFLOPS，比 oneDNN PyTorch 快 3.98x。
- **ARI 自适应指令切换**：decode（GEMV，ARI ≤ 4 token/expert）AMX 反而有 tile 开销，切到同布局的轻量 AVX-512 kernel——decode 比纯 AMX 快 1.20x，prefill 比纯 AVX-512 快 10.81x。
- **Fused MoE + 动态任务调度**：Gate/Up/Down 跨 expert 融合成两个大批次；prefill 期 expert 负载不均，用轻量任务队列动态偷取，prefill 提速 1.83x。

### 2. 异步 CPU-GPU 调度 + NUMA 感知 TP（§3.3）
- GPU 算完 gating 后，控制线程把 routed expert 任务推进无锁队列、同时发射 GPU shared expert kernel；worker 线程并发执行。
- **关键技巧：submit/sync 两个屏障包进 `cudaLaunchHostFunc`**，回调在 CUDA stream 内触发 → 整个 decode 路径捕获进**单个** CUDA Graph，消除 host 中断，decode 提速 1.23x。（FreeToken 的 graph-resident CPU 分支用的是同一招，且把 miss 决策也搬进了图内。）
- **NUMA 感知张量并行**：不用 expert 并行（整 expert 归某 socket，负载不均），而把每个 expert 的权重矩阵按列/行切到各 socket，本地计算 + 轻量 reduce-scatter——避免跨 socket 流量，decode 提速 1.63x。

### 3. Expert Deferral（§4，论文最新颖的部分）
- 观察：混合执行时 MoE 与 attention 严格交替，CPU 等 GPU 空闲多（DS-V3 单层：CPU 利用率 74%、GPU 仅 28%，重叠只占 5%）；shared expert 只占 GPU 执行时间 18%，盖不住 CPU。
- 做法：每层 top-k 里路由分最高的 I 个 expert 立即算（immediate），其余 D 个**推迟一层**：输出不喂给 layer k+1 的 attention，而合并进 layer k+2。利用残差连接的鲁棒性，行为近似但打破层间依赖，CPU 的 deferred expert 与 GPU 的下一层 attention 并行。
- 配置启发式：defer 到刚好打满 CPU 为止，且至少保留 2 个 immediate expert；最后一层不做 deferral；**只用于 decode**（prefill 时 immediate+deferred 并集几乎覆盖全部 expert，访存足迹翻倍反而更慢）。
- 效果：DS-V3 用 5+3 配置，CPU/GPU 利用率 74%/28% → 100%/37%，单层时间 -26%，端到端 decode +33%，最高 1.45x。
- 精度代价实测：LiveBench 上默认 6 个受影响 expert 平均掉 0.5%，而直接丢弃同数量 expert（Expert Skipping）掉 13.3%——deferral 显著优于 skipping。HumanEval/MBPP/GSM8K/StrategyQA 上波动 ≤2 分。

### 4. YAML 注入框架（§5）
- 基于 HF Transformers，YAML 里 match（regex 类名/模块名）+ replace（新算子、设备、kwargs）规则驱动模块替换； Listing 1 示例：DeepseekV3MoE → FusedMoE(cpu, hybrid_AMX_AVX512, Int4, n_deferred_experts=6)，self_attn → FlashInferMLA(cuda:0)，nn.Linear → MarlinLinear(cuda:0, Int4)。
- 即本地仓库 `archive/ktransformers/optimize/` 的那套机制；现役主线已改为 kt-kernel 库 + SGLang fork 集成。

## 实验设置与结果

- 硬件：双路 Xeon Platinum 8452Y（36 核/路，每路 1TB DDR5；intra-socket 220GB/s、cross-socket 125GB/s）+ A100 40GB / RTX 4080 16GB，PCIe 4.0 32GB/s。
- 模型：DS-V3 671B（GPU 17B / CPU 654B）、DS-V2.5 236B、Qwen2-57B-A14B；A100 跑 BF16 全精度，4080 跑 INT4/INT8。
- 端到端：全精度 decode 比 Fiddler 快 2.42-4.09x、比 llama.cpp 快 1.25-1.76x（量化后 1.77-1.93x）；叠加 Expert Deferral 总加速 1.66-4.90x。prefill 4.62-19.74x。
- Breakdown：AMX 主要赢 prefill（最高 3.14x），AVX-512 赢 decode（2.22x）；动态调度赢 prefill（1.83x）；NUMA TP 赢 decode（1.63x，访存受限阶段）；CUDA Graph 只对 decode 显著（1.23x）。

## 与 FreeToken 论文/代码的对照

| 维度 | KTransformers (SOSP'25) | FreeToken (arXiv'26) |
|---|---|---|
| Expert 划分 | 静态：shared/热 expert 在 GPU，routed 全在 CPU（离线 profiling） | 动态：全量 host 池 + GPU LRU cache，逐 step 跟随路由 |
| Miss 服务 | 全部 CPU 就地算（PCIe 只搬激活） | q\* 闭式切分：PCIe 取回 vs CPU 就地，按实测带宽比 |
| CPU-GPU 重叠 | Expert Deferral（**近似**：结果晚一层合并，掉 0.5% 精度） | 流内握手 + shared expert 先行（**精确**） |
| CUDA Graph | submit/sync 包进 cudaLaunchHostFunc，单图 | 连 miss 检测/切分/victim 选择都做成设备驻留数据入图 |
| CPU kernel 重点 | AMX（服务器至强）+ AVX-512 自适应 + NUMA TP | AVX-512 BF16 分派，无 AMX（面向消费 CPU） |
| 评测硬件 | 双路至强 440GB/s + A100/4080 | 消费机为主，服务器 CPU 压到 6-8 线程模拟边缘 |
| Agentic 负载 | 未涉及（Wikitext prompt） | 核心卖点（语义锚点 checkpoint、尾部 TTFT） |

## 注意点 / 批判性阅读

1. **两套论文的互评都要打折**：KT 论文（2025 年初投稿）的 baseline 没有 FreeToken；FreeToken 论文评测 KT 时把 CPU 压到 6-8 线程，KT 的 AMX/NUMA 优势（其主场是双路至强 440GB/s）被完全封印。两者实际上针对不同的硬件甜蜜点：**KT 适合"强 CPU + 一张卡"的工作站/服务器，FreeToken 适合"普通桌面 CPU + 游戏卡"**。
2. Expert Deferral 是精度-性能交换（论文自己量化并承认），FreeToken 把"精确执行"作为设计红线明确批评了这类近似——选型时这是真实的 trade-off 分歧点。
3. KT 论文的 inject 框架已在仓库里进 `archive/`，现役 kt-kernel 把 MoE 分流做成 SGLang 插件；论文里的 balance_serve/kvc2 等 serving 组件是另一条线（`archive/csrc/balance_serve/`）。
4. 两者都独立发现并使用了同一个 trick（cudaLaunchHostFunc 把 host 回调嵌进 CUDA graph），说明"图内异构调度"是该方向的公认工程分水岭。
