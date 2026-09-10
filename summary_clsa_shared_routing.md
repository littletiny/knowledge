# You Only Index Once: Cross-Layer Sparse Attention with Shared Routing (CLSA)

- 论文:https://arxiv.org/abs/2606.06467 (2026-06,Microsoft Research + Tsinghua;Yutao Sun, Yanqi Zhang, Li Dong, Jianyong Wang, Furu Wei)
- TeX 源码缓存:`~/.cache/nanochat/knowledge/2606.06467/`
- 一句话:在 YOCO 式跨层 KV 共享架构上,把"共享"从 KV cache 进一步扩展到 **top-k 路由索引**——一个单头 indexer 只算一次 token 级 top-k,所有 cross-decoder 层复用同一份稀疏索引,从而同时改善 prefill、KV cache 存储、decode 三大瓶颈。128K 上下文下 decode 提速最高 7.6×,端到端吞吐最高 17.1×(B200,vLLM 集成)。

## 1. 问题动机

长上下文推理(尤其 reasoning/CoT 场景)越来越被 decode 效率约束。现有稀疏注意力的两难:

- **Block sparse**(NSA、MoBA、Quest 等):结构化、对 GPU 友好、加速大,但块级归纳偏置粗,质量损失明显。邻近 token 语义角色可能完全不同,一个 block 里可能混着无关 token 和关键 token。
- **Token sparse**(DSA / DeepSeek-V3.2):粒度细、更准,但每层都要在**全量 cache** 上做一次 top-k 路由。top-k 是不规则算子、吃不了 Tensor Core,wall-clock 上可与大段 dense attention 相当;逐层重复计算使端到端加速有限。论文实测:128K 时 DSA 单层延迟**比 dense Transformer 还慢**,因为未摊销的 top-k 主导了成本。

## 2. 方法

### 2.1 架构(基于 YOCO)

YOCO 把模型分成 self-decoder + cross-decoder:self-decoder 用 SWA(本文 window=512)编码输入并**只构建一份共享 KV cache**;cross-decoder 各层通过 cross-attention 读这份共享 cache。

CLSA 的改动:在共享 hidden states $H$ 上加一个**单头** indexer 分支:

$$Q_{idx} = H W^Q_{idx},\quad K_{idx} = H W^K_{idx},\quad I = Q_{idx} K_{idx}^\top,\quad S_t = \mathrm{TopK}(I_t, k)$$

所有 cross-decoder 层复用同一份 $S_t$:$O_t^{(l)} = \mathrm{Attn}(Q_t^{(l)}, K_{S_t}, V_{S_t})$。注意 layer query $Q^{(l)}$ 仍然每层独立,FFN 也不变——**只有路由索引被共享**。核心原则:当多层读同一份 memory 时,路由决策也应绑定到这份 memory 并只算一次。

### 2.2 多层蒸馏训练 indexer

共享索引必须对**所有层同时有用**,而非匹配单层偏好。用 dense cross-attention 的全体层/头的注意力分布平均作为蒸馏目标:

$$\bar{A} = \frac{1}{LH}\sum_{l,h}\mathrm{softmax}(Q^{(l,h)}K^{(h)\top}),\quad \mathcal{L}_{KD} = \frac{1}{n}\sum_t \mathrm{KL}(\mathrm{sg}[\bar{A}_t] \| \mathrm{softmax}(I_t))$$

两阶段训练:
1. **Stage 1(indexer warmup)**:冻结 backbone,只训 $\mathcal{L}_{KD}$,先让路由模式稳定。
2. **Stage 2(joint sparse adaptation)**:$\mathcal{L}_{LM} + 0.1 \cdot \mathcal{L}_{KD}$,让 backbone 适应稀疏注意力分布。

### 2.3 复杂度对比(论文 Table 1)

| 模型 | KV Cache | Prefill | Decode |
|---|---|---|---|
| Transformer | $O(LND)$ | $O(LN^2D)$ | $O(LND)$ |
| YOCO (Dense) | $O((N+W_1L)D)$ | $O(\frac{L}{2}W_1ND)$ | $O(\frac{L}{2}(N+W_1)D)$ |
| DSA | $O(LND)$ | $O(LW_2ND+\eta LN^2)$ | $O(LW_2D+\eta LN)$ ← indexer 项 $\eta LN$ 逐层付 |
| **YOCO (CLSA)** | $O((N+W_1L)D)$ | $O(\frac{L}{2}W_1ND)$ | $O(\frac{L}{2}(W_1+W_2)D + \eta N)$ ← **$\eta N$ 只付一次** |

关键点:DSA 只减少每层读取的 token 数,KV cache 仍是 $O(LND)$,不解决显存瓶颈;CLSA 通过 YOCO 共享 cache 解决存储,通过共享索引把昂贵的 top-k 成本摊销到整个 cross-decoder 栈。

## 3. 实验

**设置**:4B 规模,32 层(YOCO 变体 16 self + 16 cross),hidden 2560,20 头 / 4 KV 头,head dim 128,QK norm,GQA。Transformer 用 RoPE(base=5e5);YOCO 用 RNoPE(SWA 内 RoPE base=1e4,全局 attention 用 NoPE)。CLSA 最大激活 token 数 $k=2048$(32K 训练长度下即 1:16 激活率)。Dense pretrain:8K→32K 两阶段共 135K 步;sparse adaptation 仅 2×2500 步(很轻量)。

**质量**:
- 通用 benchmark(ARC-C/BBH/GSM8K/HellaSwag/HumanEval/MMLU/DROP/WinoGrande):CLSA 与 dense 基线持平,ARC-C、GSM8K、DROP 还略有提升。
- RULER:16K 时 TRM 最好(64.4)但 CLSA(62.9)≈ dense YOCO(62.7);**32K 时 CLSA 最高(53.1)**,优势主要来自 multi-needle(MK1/MK2)。
- 长文验证 loss(Books/ArXiv/StarCoder,8K–32K):CLSA 与 dense YOCO 曲线几乎重合,基本无损。
- 稀疏度分析:k=2048 时已覆盖约 80% dense attention 质量;loss 与 dense 差距 ≤0.006(StarCoder 上甚至略优)。说明**恢复 100% attention mass 不是保住质量必要条件**。

**效率**(B200,vLLM 集成,appendix 原始数据):
- Decode tok/s @128K:Transformer 431 → dense YOCO 961 → **CLSA 3277**(7.6× TRM,3.4× dense YOCO)。
- Prefill @128K:TRM 1019 → CLSA 20742(≈20×,继承 YOCO)。
- 端到端 @128K:62.5 → 1068 tok/s(**17.1×**)。
- 单层延迟分解 @128K:TRM attention 2.11ms;CLSA sparse attn 0.05ms + 摊销 top-k 0.08ms/层(一次性 top-k 成本除以 32 层归一化,实际由 16 个 cross-decoder 层共享)。**未摊销的 top-k 可比 dense attention 还贵**——这是全文的经验核心:top-k 不是 FLOPs 问题,是 GPU 执行模式问题。
- 与 DSA、IndexCache(4 层共享索引)、HySparse(block-sparse+dense 混合)对比:128K 单层延迟 CLSA 最低。

## 4. 相关工作定位

- 跨层 salient token 复用并非全新观察:training-free 的 Kascade/OmniKV/TidalDecode,training-aware 的 HySparse/IndexCache 都利用过"相邻层显著 token 稳定"这一现象。CLSA 的差异在于把它**嵌入 YOCO 架构**,使共享索引与共享 KV cache 天然对齐(共享的有效性来自共享 KV memory 诱导的跨层注意力相似性,而非外加的 block 结构),且 prefill/存储/decode 三者同时受益。
- 与 hybrid 架构(Mamba/Gated DeltaNet + softmax 混合)正交。

## 5. 与本工作区源码的联系

- **DSA(DeepSeek Sparse Attention)** 是 CLSA 的直接对照:DSA 每层独立做 token 级 top-k,decode 成本含 $\eta LN$ 项。本机 `~/source_code/vllm/`、`~/source_code/sglang/`、`~/source_code/ktransformers/` 中均有 DSA/lightning indexer 相关实现(参见 `~/knowledge/summary_dpskv32.md`、`summary_dpskv4.md`)。CLSA 的论点——**逐层 top-k 在 GPU 上的 wall-clock 成本可抵消稀疏化收益**——对评估 vLLM/SGLang 中 DSA 的实际收益有直接参考意义:若 indexer 不能被摊销或融合进 kernel,token-sparse 在长上下文 decode 下未必赢 dense。
- **vLLM 集成路径**:论文直接把 CLSA 合入 vLLM 报告端到端吞吐,说明新稀疏注意力架构落地的标准动作就是改 vLLM 的 attention backend/model executor;`~/source_code/vllm/` 是验证此类架构 serving 行为的对照代码库。
- **YOCO / 跨层 KV 共享**:YOCO 的 self/cross-decoder 划分与 CLA、Gemma 3n、Phi-4-mini-Flash 的 KV 共享一脉相承。CLSA 说明共享粒度可以从 KV 延伸到 routing index——对设计 KV cache 管理(mooncake、FreeToken 等)有启发:**索引/路由元数据本身也是可以跨层共享、可缓存的对象**。
- **训练成本低**:sparse adaptation 只需 5000 步(对比 dense pretrain 135K 步),且 indexer 为单头小投影。这种"dense 预训练 + 轻量稀疏适配"的范式(同 NSA/DSA 的后训练路线)值得在做 sparse attention 实验时复用。
- **可借鉴的评测方法**:attention coverage(top-k 集合覆盖 dense attention mass 的比例)与 CE loss 联合分析,说明"覆盖率 ≠ 质量",做稀疏注意力消融时应有这两个维度。

## 6. 局限与开放问题

- 仅在 4B、训练上下文 32K 上验证,128K 效率数字来自 serving 外推(RULER 只测到 32K);更大模型/更长训练上下文下共享索引是否仍无损未知。
- indexer 的 top-k 在 prefill 阶段仍需对每个 query 位置计算($\eta N$ 一次),prefill 的主要收益仍来自 YOCO 而非 CLSA。
- $k=2048$ 固定;超长上下文下固定 k 的覆盖率会下降,是否需要自适应 k 未讨论。
- 共享索引绑定共享 KV memory——该技巧依赖 YOCO 式架构,不能直接迁移到普通逐层 KV 的模型(对普通架构只能用 IndexCache 式的近似共享)。
