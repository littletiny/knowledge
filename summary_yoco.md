# YOCO: You Only Cache Once — Decoder-Decoder Architectures for Language Models

- **arXiv**: 2405.05254(NeurIPS 2024,Microsoft Research + 清华;Yutao Sun、Li Dong、Furu Wei 等)
- **代码**: https://aka.ms/YOCO (microsoft/unilm)
- **TeX 源码缓存**: `~/.cache/nanochat/knowledge/2405.05254/`
- **相关阅读**: 与 `summary_random_attention.md`、`summary_dpskv32.md` 等 KV cache 优化类总结对照

## 一句话总结

把 LLM 拆成 **self-decoder(前半 L/2 层,用 O(1) 推理内存的高效注意力)** + **cross-decoder(后半 L/2 层,用 cross-attention 复用 self-decoder 输出产生的、全局唯一一份 KV cache)**,整体对外行为等价于 decoder-only Transformer;KV cache 内存降约 L 倍(65B 模型 ~80×),prefill 可在进入 cross-decoder 前早退(1M 长度 prefill 加速 71.8×,512K 从 180s 降到 <6s),且不损失 scaling 能力与长上下文检索精度(3B 模型扩展到 1M 上下文,needle 检索近满分)。

## 动机

1. **KV cache 是内存瓶颈**:65B 模型(GQA + 8bit KV 量化)存 512K token 的 KV 需 ~86GB,超过单卡 H100。
2. **Prefill 延迟随长度平方增长**:4×H100 上 7B 模型 prefill 450K token 要 ~110s,1M 要 ~380s。
3. 现有高效架构(SSM/线性注意力)省内存但全局检索能力弱;YOCO 的思路是**在架构层面把"全局记忆"只存一份**。

## 架构

```
X^0 → [Self-Decoder × L/2] → M = X^{L/2} → K̂ = LN(M)·W_K, V̂ = LN(M)·W_V
                                        ↘ [Cross-Decoder × L/2]: Attention(Q^l, K̂, V̂) → X^L → softmax
```

- **Self-decoder**:block 结构与 Transformer 相同(pre-RMSNorm + SwiGLU),但 attention 换成 **efficient self-attention(ESA)**,要求 O(1) 推理内存。论文给出两种实例:
  - **Gated Retention(gRet, 默认)**:RetNet + 数据依赖门控衰减 γ = sigmoid(X·W_γ)^{1/τ}(τ 温度项鼓励 γ→1 利于记忆)。**head-wise 而非 element-wise 衰减**,为了能用 tensor core 打满(对比 GLA 的 element-wise decay)。支持 parallel / recurrent / chunkwise 三种等价表示(附录有 chunkwise 与 recurrent 等价性证明)。
  - **Sliding-Window Attention(SWA)**:窗口 C=1024,cache 大小只与窗口有关。
- **Cross-decoder**:K̂/V̂ 由 self-decoder 最后一层输出一次性算出,被**所有** L/2 个 cross-decoder 层复用(GQA 兼容,可进一步省)。每层只学自己的 W_Q^l。因果 mask 保持自回归。
- 3B 配置:d=3072,L=26,Q heads=24,KV heads=8,tiktoken cl100k,训练 1.6T token(5T token schedule)。

### 推理复杂度对比(N=长度,L=层数,D=维度)

| | KV cache 内存 | Prefill 时间(attention 部分) |
|---|---|---|
| Transformer | O(LND) | O(LN²D) |
| YOCO | O((N+L)D) ≈ 一份全局 KV + 常数 | O(LND)(线性) |

- **Prefill 早退**:cross-decoder 只依赖 K̂/V̂,而 decode 第一个 token 只需 self-decoder 算完 + 最后一层 cross-attention 对最后一个位置,所以 prefill 可以提前退出 cross-decoder 的大部分计算 → 至少 2× 加速,再叠加 ESA 本身的线性复杂度。

## 实验要点

- **Scaling(160M→13B,各 10B token)**:YOCO_gRet 验证 loss 曲线与 Llama 式 Transformer 持平且略优;gRet > SWA;**attention/retention 交错(1:3)的混合架构互补性最好**(与 Jamba 发现一致)。
- **3B + 1.6T token**:LM Eval Harness 平均分 0.636,超过同量级 OpenLLaMA-3B-v2(0.619)/ StableLM-3B-4E1T;扩到 1M 上下文后(YOCO-3B-1M)继续微升至 0.645。
- **长上下文**:渐进式扩长 64K→256K→1M(RoPE θ: 640K→5M→80M,每阶段 token 6B/4B/1.5B);单针检索近满分;多针(128K,N=1/2/4/8:0.98/0.98/0.84/0.56)优于 ChatGLM3-128K、YaRN-Mistral,与 7B 的 LWM-1M 相当(参数量只有一半);book/repo-code 的累计 NLL 随长度幂律下降,说明真用上了长程信息。
- **推理 profiling(H100,chunk size 256,Triton kernel 基于 FLA)**:1M 长度总内存 12.4GB vs Transformer 9.4×;65B 模型 1GB 内存可服务 128K token(Transformer+GQA 只能 1.6K);prefill 512K 从 180s→<6s;吞吐 512K 下 43.1 vs 4.5 token/s(9.6×)。
- **消融(160M,Zoology AR-Hit/First-Occur 细粒度 ppl)**:YOCO_gRet 在 AR-Hit(联想回忆)上甚至优于 Transformer(1.199 vs 1.219),远好于 Mamba/RetNet/H3 —— 说明"全局 cross-attention + 局部 retention"的分工保住了召回能力;ZeroSCROLLS 长文任务上 YOCO 与 Transformer 第一梯队。
- **Chunk parallelism(附录)**:序列切到多卡后,self-decoder 只有相邻卡通信(传 recurrent state S_n 或 SWA 窗口),cross-decoder 的 KV 只需 **all-gather 一次**而非每层一次 → 长序列训练的通信开销和显存碎片都更小。

## 值得注意的设计细节

- 全局 KV 由 **LN(M)·W_K/W_V** 生成:K/V 投影前有一个额外的 LayerNorm。
- gRet 的三种范式:训练/prefill 用 chunkwise(chunk 内并行 + chunk 间递归),decode 用 recurrent(state S_n ∈ R^{d×d},O(1) 内存),伪代码见附录。数值上 γ 用 logsigmoid 累乘再 exp。
- Multi-head gRet 用 GroupNorm 逐头归一化 + swish 门(沿用 RetNet/Magneto 设计)。
- FFN 用 3d(而非 Llama 的 8/3 d)来对齐参数量。
- 结论部分展望:YOCO + BitNet + Groq(权重和 KV 都极致压缩后全放 SRAM);多模态(多个 self-decoder + cross-attention 天然是模态融合);**把全局 KV cache 当作一等公民做压缩/索引/预缓存(native RAG)**——因为只缓存一次,只需要维护一份索引。

## 与 ~/source_code/ 生态的联系

YOCO 的核心思想在今天(2026)已经广泛落地为 **"hybrid 架构"**:大部分层用线性注意力/SSM/SWA(O(1) 或 O(C) 状态),少量层保留 full attention 保全局检索。相关实现:

- **sglang**:`sglang/python/sglang/srt/layers/attention/linear/` 下的 `gdn_backend.py`(Gated DeltaNet,Qwen3-Next/Kimi Linear 同款数据依赖门控线性注意力,gRet 的直接后继)、`kda_backend.py`(Kimi Delta Attention)、`lightning_backend.py` 等就是 gRet 这类 chunkwise 线性注意力的生产实现;`mem_cache/hybrid_cache/`、`unified_memory_pool.py` 处理 hybrid 模型的"full-attn 层 KV pool + 线性层 state pool"混合显存管理,`configs/hybrid_arch.py` 定义 hybrid 层型配置。YOCO 相当于 hybrid 的极端形态:**full-attention 的 KV 只有一份且挂在中间层**,意味着如果这类架构流行,KV pool 的 per-layer 管理可以大幅简化。
- **vllm**:`vllm/v1/core/kv_cache_utils.py` 的 hybrid KV cache manager(Mamba/线性层与 full-attn 层分组管理)、`tests/v1/kv_connector/unit/test_mooncake_connector_hybrid_mamba.py` 说明 KV 传输层也要感知 hybrid 层型。YOCO 提示:对"只有一份全局 KV"的模型,**KV 传输/ offload / prefix cache 的成本都降 L 倍**,对 mooncake 这类分布式 KV cache 系统(`mooncake/`)是架构级红利。
- **FreeToken**:`FreeToken/tests/kvcache/test_hybrid_swa_kv_cache.py`、`test_swa_paged_pool.py` 已实现 SWA + full-attn 混合的分页 KV 池——正是 YOCO_SWA 变体所需的推理基础设施。
- **kernel 侧**:gRet 的 chunkwise kernel 基于 FLA(flash-linear-attention);`DeepGEMM/` 的 FP8/FP4 GEMM 对线性注意力里的 rank-1 外积累加不直接适用,这类 kernel 仍走 Triton/FLA 路线。
- **启发**:如果做长上下文 serving,YOCO 证明了"prefill 早退"这一架构特性可以把 TTFT 压一个数量级以上——这是纯系统优化(prefix cache、chunked prefill)拿不到的收益,属于**架构与系统 co-design** 的范例;其"全局 KV 只一份 → 单索引 native RAG"的展望与 CacheBlend/LMCache 的 chunk 级 KV 复用思路互补。
