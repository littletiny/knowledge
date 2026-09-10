# HySparse: A Hybrid Sparse Attention Architecture with Oracle Token Selection and KV Cache Sharing

- **arXiv**: 2602.03560(2026-02,LLM-Core Xiaomi + 北京大学)
- **TeX 源码缓存**: `~/.cache/nanochat/knowledge/2602.03560/`
- **结合阅读**: `discussion_kv_selection_signals.md`(选择信号来源三部曲:CacheBlend/DA/RA)——HySparse 是对同一个问题"attention 该看谁、信号从哪来"的**第四种答案:信号来自上一层 full attention(oracle)**,且它是**训练进架构**的方案,而非推理期 trick

## 一句话总结

把 Transformer 改造成"1 层 full attention + N 层 sparse attention"的混合块:**full attention 层顺带输出 block 级注意力分数,后续 N 个 sparse 层直接复用它的 TopK token 索引和 KV cache**(sparse 层自己再带一个独立 KV 的 SWA 分支,两路输出经 sigmoid gate 融合)。由此同时解决 sparse attention 的两个老问题:(1) token 选择不再依赖 proxy(learnable indexer / 启发式),而是用 full attention 做 oracle;(2) 动态稀疏省计算不省 KV 的问题被跨层 KV 共享解决。80B-A3B MoE 49 层里**只用 5 层 full attention(1:11)**,KV cache 降近 10×,通用 benchmark 反而多数超过 Full-Attn baseline,RULER 32k 上 87.4 vs Full-Attn 82.1。

## 动机:sparse attention 的两个根本缺陷

1. **Proxy-based 选择**:training-free(StreamingLLM/H2O/Quest)靠启发式;trainable(DSA/NSA/MoBA/SeerAttention)要额外训练 indexer/gating,本身是近似、增加训练复杂度,且选择质量被 proxy 保真度 bound 住。
2. **省算不省显存**:动态稀疏(dynamic sparse)为了不掉点通常保留完整 KV cache(eviction 不可逆、token 重要性随 decode 漂移),KV 显存仍是长上下文 serving 吞吐的瓶颈。

两个被利用的经验观察(都有前作支撑):
- **跨层 salient token 稳定性**:相邻层的 heavy-hitter token 高度重合(TidalDecode、OmniKV、Kascade 等,此前只被用作 training-free 推理加速)。HySparse 把这个观察**提升为预训练架构**。
- **跨层 KV 共享无损**(YOCO、CLA、MiniCache:相邻层 KV 相似度高)。

## 方法

### 结构
重复的 hybrid block = 1 × full attention + N × sparse attention(7B dense:1:3,36 层;80B MoE:1:11,49 层)。最后一层恒为 full attention(保全局聚合)。

### Full attention 层:顺手产出 oracle 分数
- FlashAttention 的 online softmax 本来就计算 rowmax;**改造 kernel,把每个 K-tile 的 rowmax 存下来再 rescale**,得到 block 级 max attention score `S ∈ R^{t × ⌈t/B⌉}`(B=64),开销可忽略(Algorithm 1,类似 SeerAttention 的做法)。
- 对 S 做 TopK 选 block 索引 I(默认选 1024 token = 16 个 block);GQA 下同一 KV group 内的 query head 取 group-wise max,共享同一组稀疏索引(利于 kernel 效率)。
- 注意:这个分数是 softmax 归一化后的真值(除以 ℓ),不是 logits——所以它真的是"oracle",来自模型自己的注意力分布。

### Sparse 层:双分支 + 门控
- **Block sparse 分支(global)**:query 自己算(W_{q'}),但 **K/V 和 TopK 索引全部来自前面的 full attention 层**——本层零额外 KV cache。
- **SWA 分支(local)**:窗口 w=128,**有自己独立的小 KV cache** 和独立 QKV 投影;带 gpt-oss 式 per-head learnable sink bias。
- **融合**:两路输出各过一个 sigmoid gate(`σ(W_g x)`,Qwen 式 gated attention)后相加。

### 训练
预训练直接以该架构训练(非后改造)。7B:1T tokens@8K + 200B@32K 长上下文扩展(RoPE base 640K);80B MoE:500B tokens@32K。端到端,无需蒸馏/辅助 loss——因为根本没有需要训练的 selector。

## 实验结果

- **7B dense(1T tokens)**:HySparse 在 MMLU(58.8 vs 56.9)、MMLU-Pro(29.0 vs 26.8)、GSM8K(37.9 vs 33.3)、C-Eval/CMMLU 等多数项超过 Full-Attn;Hybrid SWA 大致持平但长上下文弱。
- **80B MoE(500B tokens,1:11)**:Hybrid SWA 在激进比例下明显崩(MMLU 54.9 vs Full 61.8);HySparse 几乎全面反超 Full-Attn(MMLU 62.2、GSM8K 54.1、HumanEval 38.4 vs 35.4),仅 MMLU-Pro/DROP/ARC-C 略低。**49 层仅 5 层 full attention,KV cache ~10× 缩减**。
- **RULER**:7B@32k 89.3(Full 88.2 / SWA 84.2);80B@32k **87.4 反超 Full-Attn 的 82.1**(Hybrid SWA 只有 69.5),MK3/VT 等多 key 检索项大幅恢复。
- **消融(7B)**:
  - 去掉 sparse 层内的 SWA 分支 → 全面掉点(GSM8K 37.7→29.7,DROP 52.2→46.4):即使有了 oracle 全局检索,**独立的局部通路仍不可省**。
  - SWA 分支也共享 full 层 KV(而不是独立 KV)→ 严重掉点(MMLU 58.4→52.8):SWA 需要自己的局部表征,共享全局 KV 会把局部通路"污染"成检索导向。**结论:SA 共享 KV 安全,SWA 必须自有 KV**。

## 关键洞察与设计取舍

- **选择信号的"第四种来源"**(接 `discussion_kv_selection_signals.md` 的框架):RA 用零信息随机、DA 从模型文本声明、DSV4 indexer 用学出来的 proxy,而 HySparse 用**上一层 full attention 的真实分布**——这是信号质量的天然上界(对那一层附近的层而言),且零训练成本、零额外模块。代价是:每 N 层必须付一次 O(n²) 全注意力(讨论节也承认:实践中没人能彻底消灭 O(n²),关键只是把 full:cheap 的比例压到极限——HySparse 把已知极限从 Gemma3/GPT-OSS 的 ~1:3~1:6 推到 **1:11**)。
- **与 RA 结论的张力与和解**:RA(discussion 三部曲)说 decode 驱逐时排序信号几乎不值钱;HySparse 说训练进架构时 oracle 信号值钱(反超 full)。两者不矛盾——RA 是短 prompt+长 CoT 的 eviction regime(复述冗余兜底),HySparse 是长 context 检索 regime(RULER MK3/CWE 正是 RA 会死的 needle 场景),且 HySparse 的选择**可逆**(每步重选,不是驱逐)。
- **oracle 有效性的边界**:索引复用依赖"跨 N 层 salient 稳定",N=11 时最远层用的是 11 层前的选择,这大概是它能成立且仍需最后一层 full attention 兜底的原因。
- **SWA 自有 KV 的教训**有普适性:任何"全局检索 + 局部窗口"双路设计(KDA/GDN 混合、DSV4 的 SWA+sparse),两路的表征需求不同,共享 KV 会伤局部通路。

## 系统侧含义与对本工作区的落点

- **KV offload 流水线**:论文讨论节明确提出——full 层 KV 可 offload 到外部内存、预取回 GPU;GPU 上只常驻 sparse 选中部分 + SWA 小窗。这正是 **Mooncake**(`~/source_code/mooncake/`)擅长的分层 KV 存储/传输场景:HySparse 的"1:11"结构意味着每个 block 只有一份 full KV 需要跨层传输,offload 的带宽压力天然降一个量级。OmniKV 已验证 post-training 版本的类似 offload。
- **kernel 落点**:Algorithm 1 的"FlashAttention 顺带输出 block 级 rowmax 分数"改动极小——可对照 `~/source_code/llama.cpp/ggml/src/ggml-cuda/lightning-indexer.cu`(DSA lightning indexer,proxy 打分的实现)看 HySparse 方案省掉了什么;vLLM 侧的对应物是 `~/source_code/vllm/vllm/v1/attention/backends/mla/indexer.py`(DSA indexer backend)。若要在 vLLM/SGLang 支持此类架构,核心工作是:block 表按 hybrid block 分组共享、FA kernel 增加 score 输出、sparse gather 复用现有 paged kernel。
- **与 DSA/DSV4(`summary_dpskv4.md`)的对比**:DSV4 每层都要跑 lightning indexer 做 O(N) 打分、KV 每层独立存储(压缩);HySparse 每 12 层只付一次 full attention,sparse 层零打分成本、零 KV 存储。但 DSA 的选择是**每层每步 query-dependent** 的,HySparse 的选择被冻结在 block 首层——细粒度 vs 成本的取舍。
- **架构趋势注脚**:与 MiMo-V2-Flash(同出处 Xiaomi 系)、Gemma3、GPT-OSS 的 hybrid SWA 一脉相承,HySparse 证明 hybrid 比例的极限可以推得更远;可作为评估未来模型 config(`~/model-knowledge/configs/`)是否采用类似结构的参照。
