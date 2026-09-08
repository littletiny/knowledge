# CacheBlend: Fast LLM Serving for RAG with Cached Knowledge Fusion

- **arXiv**: 2405.16444(EuroSys '25,UIUC,Junchen Jiang 组;Jiayi Yao 等)
- **代码**: https://github.com/LMCache/LMCache(CacheBlend 已并入 LMCache)
- **TeX 源码缓存**: `~/.cache/nanochat/knowledge/2405.16444/`
- **结合阅读**: 与 `summary_declarative_attention.md`(arXiv 2609.02737,Declarative Attention / DA)对照

## 一句话总结

RAG 场景下输入由**多个检索到的 text chunk** 拼成,prefix caching 只能复用第一个 chunk 的 KV,full KV reuse(PromptCache)直接拼接各 chunk 独立预计算的 KV 会**丢失 chunk 间的 cross-attention** 导致质量下降;CacheBlend 复用所有 chunk 的预计算 KV,但**每层只选择性重算 ~5–18%(默认 15%)KV 偏差最大的 token(HKVD tokens)**,恢复 cross-attention,TTFT 降 2.2–3.3×、吞吐升 2.8–5×,质量损失 ≤0.02 F1/Rouge-L。

## 问题分解

1. **Prefill 是 RAG 的瓶颈**:4K token 输入在 A40 上 prefill 需 3s(Llama-34B)/ 5.8s(Llama-70B),决定 TTFT;DistServe 显示去掉 prefill 吞吐可翻倍。
2. **Prefix caching 不够**:多 chunk 输入里只有第一个 chunk 是 prefix,其余 chunk 的 KV 无法复用。
3. **Full KV reuse 不够**:各 chunk 的 KV 是独立预计算的,**chunk 间的 cross-attention 从未被计算过**(预计算时不知道前序文本是什么),导致 forward attention 矩阵偏差、回答错误(论文用"梅西 vs C 罗世界杯进球数"的例子展示:两段生涯数据分别检索,需要跨 chunk 比较)。chunk 越多,质量差距越大。

## 方法:Selective KV Recompute

### 两个经验 insight

- **Insight 1**:重算 KV 偏差(KVD,该 token 当前 KV 与 full-prefill KV 的绝对差)最高的 token,对降低 attention 偏差(forward attention 矩阵与 full prefill 的 L2 距离)收益最大。且由于 **attention 稀疏性**,只有 ~10–15% 的 token 有高 KV 偏差 —— 它们就是与其他 chunk 有强 cross-attention 的 token。
- **Insight 2**:**相邻层的 HKVD token 集合高度相关**(Spearman rank correlation 持续很高;直觉:token 的输入 embedding 在层间变化缓慢,KV 是 embedding 的线性变换)。

### 算法

- **位置修正**:RoPE 下只需对预计算 K 向量乘一个旋转矩阵(附录证明 attention score 只依赖相对位置),开销可忽略 → 同一 chunk 的 KV 可在任意位置复用。
- **Partial prefill(逐层)**:每层只把选中的 r% token 输入 Q/K/V 变换;K/V 与未选中 token 的缓存 KV 拼接,使注意力仍覆盖全部前序 token;计算量 = r% × full prefill。
- **Gradual filtering 选 token**:第一层全量 prefill,按 token 级 KV 偏差选 r₁%(略大于目标 r)作为第二层的重算集;第二层重算后再筛出 r₂% < r₁%……逐层收窄。比"只看第一层"在统计上更可靠,尤其深层。

### 系统:用流水线把重算藏进加载延迟

- **基本洞察**:如果一层的 selective recompute 比从存储设备加载该层 KV 更快,两者流水线化后 recompute 不增加 TTFT。例:Llama-7B + 4K context,重算 15% 每层 3ms,NVMe SSD 加载一层 KV 需 16ms → 完全掩盖;因此 KV 可以放在**更慢更便宜的存储**(SSD 甚至 S3 对象存储)而不增加延迟,大幅提升可缓存的 chunk 数量和命中率。
- **Loading Controller**:两个延迟估计器(recompute delay = r% × 离线 profile 的 prefill 时间;load delay = 每 token KV 大小 × L / 设备吞吐)选出"刚好能被加载掩盖"的 recompute ratio r(且不低于经验下限 r*≈15%),并选满足 `T_recompute ≥ T_load` 的最便宜存储设备。
- **KV cache store**:按应用语义切 chunk,hash(同 vLLM block hashing)查找,LRU 驱逐;新 chunk 的 KV 异步写回磁盘。
- **实现**:vLLM 上 ~3K 行 Python;三个接口 `fetch_kv / prefill_layer / synchronize`,双线程流水线(第 i 层 recompute 与第 i+1 层 KV 加载并行)。

## 实验

- 模型:Mistral-7B / Yi-34B / Llama-70B(后两者 8-bit 量化);硬件:A40 + 128GB RAM + NVMe SSD(4.8 GB/s)。
- 数据集:2WikiMQA、Musique(多跳 QA,F1)、SAMSum、MultiNews(摘要,Rouge-L);另构建 Musique/2WikiMQA extended(GPT-4 扩写 query,top-6 × 512-token chunk)模拟 RAG 复用负载。
- 结果:vs full KV recompute,TTFT ↓2.2–3.3×,质量损失 ≤0.02;vs full KV reuse,F1/Rouge-L 高 0.15–0.35(很多情况 2× 以上);同 TTFT 下吞吐 ↑2.8–5×。recompute ratio 5–18% 时质量损失最多 0.002。
- 局限(作者自述):仅 transformer 架构;只测了三个模型、A40 时代硬件;chunk 512 token、context ~4K 规模;未测跨节点 KV 共享。

---

## 与 Declarative Attention(2609.02737)的结合分析

两篇论文处理的是**同一个根本矛盾的两端**:长上下文/多 chunk 输入下,模型实际需要的信息只是很小一部分,但默认实现要为全部上下文付全价。它们恰好覆盖 inference 的两个阶段:

| 维度 | CacheBlend (2405.16444) | DA (2609.02737) |
|---|---|---|
| 优化阶段 | **Prefill**(TTFT) | **Decode**(每步 KV 读带宽) |
| 复用/节省对象 | 跨请求复用 chunk 的预计算 KV,只重算 15% | 请求内每步只读声明的 KV 段 |
| 选择机制 | **extrinsic**:实测 KV 偏差,选 HKVD token(token 级、逐层) | **intrinsic**:模型在 CoT 里用文本声明(chunk 级、按 reasoning span) |
| 粒度 | token 级(每层 r% 个 token) | segment 级(~2K token 的 magic chunk,block 对齐) |
| 质量保障 | 恢复 cross-attention(逼近 full prefill) | mask 可逆、不驱逐,随时可回 global 全看 |
| 时代/规模假设 | 2024:~4K context,A40,prefill 主导 | 2026:100K–1M context,B200,decode attention 主导 |

### 互补点(可以直接拼成一条流水线)

1. **阶段互补**:CacheBlend 省 prefill,DA 省 decode,合在一起覆盖整个请求生命周期。在 DA 论文的 roofline 框架里,prefill 被显式排除(PD 分离下到另一个 pool)——CacheBlend 正是压那个 pool 成本的技术。
2. **DA 的 magic chunk 就是 RAG chunk**:DA 把 context 伪装成"已逐段检索完毕的 tool-use transcript";真实 RAG/agentic 场景里这些段本来就来自 retriever,其 KV 可以用 LMCache/CacheBlend 方式**离线预计算、落盘、按需加载**。组合形态:chunk KV 预计算+存储(LMCache)→ 组装成 DA prompt(magic chunk transcript)→ decode 时 DA state machine 按声明 mask。
3. **KV 加载的可预测性是共同主题**:CacheBlend 用逐层流水线把 recompute 藏进加载延迟;DA 的"system-2 KV offloading / 可逆 compaction"愿景依赖 focus 声明**提前在文本中宣告**,从而能预取失焦段。两者都把"访问模式可预知"变成"可以用慢存储/隐藏延迟"。这比 Quest/DSA 类每步重选的方法更适合分层存储——Mooncake(`~/source_code/mooncake/`)的 KV 分层正是这个方向。

### 张力点(结合时要小心)

- **cross-attention vs 段级 mask**:CacheBlend 的核心教训是独立预计算的 chunk KV 缺 cross-attention 会掉质量,需要重算 15% HKVD token 来修复。DA 不做这种修复——它假设 context 是一次性 prefill 的(cross-attention 完整),decode 时只是选择不读。如果要把两者叠加(预计算 chunk KV + DA 解码),**必须先解决 chunk KV 的 cross-attention 缺失**:要么 CacheBlend 式重算,要么接受 DA 在 global 模式下对跨段推理的质量损失。DA 论文的失败案例分类(跨段计数 cwe、被切开的表格)与 CacheBlend 的 cross-attention 缺失其实是同一类问题的不同表现。
- **粒度不匹配**:CacheBlend 的选择是 token 级、逐层不同的;DA 是段级、全 global 层统一的 block mask。混合架构(SWA/GDN + 少量 global 层)下 CacheBlend 的逐层重算框架仍然成立,但 DA 只作用于 global 层。
- **规模假设差异**:CacheBlend 的 15% 重算在 ~4K context 下成立;到 100K+ context、128K+ 窗口的现代模型上,HKVD 比例和 cross-attention 分布是否仍成立需要重新验证(DA 论文附录的 KV 字节表显示现代模型的每 token KV 开销已降 1–2 个数量级,预计算 KV 的存储/加载成本结构也变了)。

### 对本工作区的具体关联

- **LMCache** 已把 CacheBlend 产品化,与 `~/source_code/vllm/`(vLLM 的 KV connector / offloading 接口)、`~/source_code/mooncake/`(KV 传输层)直接相关。
- DA 论文的 vLLM 集成(hook attention metadata builder 改 block table)与 CacheBlend 的 vLLM 集成(逐层 partial prefill 接口)都是"不改 kernel 改调度/metadata"的低成本路线,做引擎侧实验时可互相参考。
- DA 总结(`summary_declarative_attention.md`)里的"可逆 context compaction"设想,其加载端基本可以复用 CacheBlend/LMCache 的流水线加载思路。
