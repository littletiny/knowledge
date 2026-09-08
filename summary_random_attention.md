# Random Attention: Rethinking KV Cache Eviction for Efficient Reasoning

- **arXiv**: 2609.03430(2026-09,TMLR 格式)
- **TeX 源码缓存**: `~/.cache/nanochat/knowledge/2609.03430/`
- **结合阅读**: 与 `summary_declarative_attention.md`(DA,2609.02737)、`summary_cacheblend.md`(CacheBlend,2405.16444)构成同一主题三部曲:长序列下 KV cache 的"选择信号到底值多少钱"

## 一句话总结

推理模型(short prompt + 数万 token CoT)的 decode 期 KV cache 驱逐,过去一直被当作**排序问题**(给每个 token 打分留 top-K);本文直接检验这个前提,发现**选择信号几乎不贡献准确率**——只要**保住整个 prompt**,在每个 KV head 内**均匀随机驱逐**(Random Attention, RA)就与最强 baseline(TriAttention)打平甚至显著领先(60 个对比中 31 个显著占优、仅 1 个显著落后),且因为没有 scoring pass,vLLM paged serving 下吞吐高 32–43%。

## 设定与方法

- **Eviction ≠ selection**:eviction 是永久性丢弃,能真正 bound 住显存(这是推理场景的刚需);sparse-attention selection(Quest 等)cache 仍线性增长。本文只研究前者。
- **周期性驱逐框架**(沿用 R-KV/VaSE):cache = 预算 K + 最近 R=64 token 的 buffer(永不打分);每 64 步触发一次驱逐,候选集打分取 top-K。
- **各 baseline 即不同的打分函数**:StreamingLLM(sink+recency,无分)、H2O(累计注意力)、SnapKV(最近 w 个 query 的注意力)、R-KV(SnapKV + key 余弦冗余惩罚)、VaSE(value 幅度 + SnapKV 比例采样)、TriAttention(按与当前 query 的距离打分,三角级数系数按 head 校准 + ‖k‖)。
- **Random Attention 全部方法只有两步**:① prompt 位置打分 +∞(整个 prefill:system、模板、问题,永不驱逐);② 其余位置每个 KV head 独立抽 Uniform(0,1)。每次驱逐的成本 = 一次 `rand` + 一次 `topk`,四行代码。
- 隐含性质:存活概率每轮 ~0.94ⁿ,所以 RA 实际是**软 recency window + 每个 head 不同的稀薄长尾**。

## 实验

- 模型 Qwen3-4B/14B/32B + Phi-4-reasoning;任务 MATH500、GPQA、AIME 25/26、HMMT、LiveCodeBench-v6;~4× 压缩(K=1024–4096);32k 生成上限;所有结论过 paired bootstrap + 符号检验。
- 主结果:RA 在数学/科学推理上显著超过 VaSE、SnapKV(所有模型)和 R-KV(4B);无人显著超过 RA。唯一显著落后是 Qwen3-32B 的代码任务——**根因是 LCB prompt 平均 557 token(是 MATH 的 6 倍),保护整个 prompt 就吃掉一半预算**,与选择信号无关。
- 压缩压力 2×→16× 扫描:RA 与 TriAttention 始终并列,与 VaSE 的差距随压缩率拉大。
- **效率**(H200,32k 生成):vLLM paged serving 下 RA 是 full attention 的 1.6–2.7×,比 TriAttention 高 32–43%。原因:单次 scoring 虽只贵 ~1.3ms,但 128 并发下**几乎每个 decode step 都有请求在压缩,且压缩发生在 batched step 之间的同步点,全体等待**(TriAttention 每次压缩让整批等 ~15ms);且 fused kernel 不产出 attention 统计,content-dependent 打分会强制额外的 KV 扫描。等显存对比(HF 批解码)下 RA 达 3–10×(K=1024 时 28.8×)full-attention 吞吐。
- 短生成(8k)时压缩本身不划算,RA 反而 0.52–0.96×——作者如实报告。

## 机制解释(全文精华)

1. **Prompt 是 cache 里唯一脆弱的部分**:各 baseline 对 prompt 的处理本来就不同(TriAttention 默认保住,VaSE/R-KV/SnapKV 只留 sink 交给分数),所以跨论文对比其实在比"保护规则"。统一加上"保住 prompt"后:方法间差距基本消失,**每个方法的提升幅度恰好等于它的分数原本丢失 prompt 的比例**(SnapKV 最多 +22.5 分;R-KV 几乎不涨)。反过来说:不保 prompt 时 recency window 崩到 0.09、RA 掉到 0.23–0.76。**此前论文报告"随机基线远差于打分法"的混淆变量就是随机基线丢了 prompt。**
2. **推理轨迹自我保护——两级冗余**:
   - **文本级**:模型会复述自己还在用的中间结论(R-KV 已指出),所以重要值很少只存在于一个位置。
   - **跨 head 级**:每个 KV head 各存一份拷贝,per-head 独立驱逐意味着"所有 head 同时丢掉它"才真正丢失。Planted-fact probe(把合成事实 `zq = 4729` 插进真实 MATH 轨迹、只 pin 在指定 head):单 head 检索率 3%,2 head 60%,3 head 83%,8 head 99%——**跨 head 读取是超加性的**;甚至两个事实分别存在不同 head 也能 pooling。且**拷贝的形状无关紧要**:把事实按 token 轮流撒在不同 head(没有任何 head 持有可读片段)检索率几乎不掉;连续 block 到 64 token 都无损失,重要的是"每个 head 剩几个 block"而非 block 长度。
   - 真实轨迹上 cross-head 多样性甚至可以不要(shared-draw 对照组差 0.3 分以内),因为文本级冗余已经够;跨 head 级兜底的是"只说一次、从不复述"的内容。
3. **选择信号仅剩的领地**:只说一次、很久之后才用到的罕见事实(passcode probe,隔 57 轮压缩):RA 检索率 0;R-KV(全程累计注意力)84%;VaSE 1/3;SnapKV/TriAttention 几乎为 0。但 **needle 能力与聚合任务强度不相关**(R-KV 找针最强却只赢一列;TriAttention 总分最强却找不到针)。

## 实践含义

- RA 是可部署的默认方法:零校准、零超参、零 scoring pass,同准确率下最快;也是**任何新选择信号必须打败的 null hypothesis**(必须在同预算、同 prompt 保护下比)。
- 驱逐研究该优化的问题被重定向:准确率由"保护什么"决定,而非"怎么排序"。开放问题 = ① 长 prompt 的预算分配(尤其代码任务,全 pin 太浪费,应压缩脚手架而非整体 pin);② 罕见一次性事实的恢复(只有 content-dependent 信号能救)。

## 与 DA / CacheBlend 的三角关系

三篇论文构成一条清晰的线:**"attention 该看谁"的决定可以从三个来源获得——代理分数(extrinsic)、模型文本(intrinsic)、结构规则(structural)**:

| | CacheBlend (2405.16444) | DA (2609.02737) | Random Attention (2609.03430) |
|---|---|---|---|
| 阶段 | prefill(跨请求 KV 复用) | decode(每步读哪些 KV) | decode(哪些 KV 永久丢弃) |
| 选择信号 | extrinsic(KV 偏差实测) | intrinsic(模型 CoT 声明) | 无(结构规则:保 prompt + 随机) |
| 结论 | 15% token 重算即可修复 cross-attention | 声明式 mask 省 31–52% 读 | 排序信号几乎不值钱,保护才值钱 |

- **RA 与 DA 的深层呼应**:RA 发现推理轨迹靠"复述"获得文本级冗余——而 DA 的 `<focus>` 模式恰恰**制度化了这种复述**(从 chunk 里 verbatim 抽取值到 response),`<local>` 模式正是"只看自己复述出来的工作区"。可以说 DA 的协议主动制造了让随机驱逐可行的冗余结构;反过来 RA 解释了为什么 DA 的 local 模式敢完全不看 context。
- **RA 与 DA 的互补 regime**:RA 针对"短 prompt + 长生成"(prompt 脆弱);DA 针对"长 context + 推理"(context 巨大、scaffold 常看)。RA 的失效区(只说一次的 needle)正是 DA 评测的 RULER NIAH 主战场——那里 DA 的 global→focus 导航保留了找回 needle 的能力,而 RA 会永久丢失。
- **对打分派的共同批判**:RA 说 decode 期驱逐的打分不省钱也不增值;DA 说每步 O(N) 的代理扫描可以换成 O(1) 文本声明;两者都指向"选择信号应该从结构/文本里免费获得,而不是每步计算"。CacheBlend 的 HKVD 打分能成立,是因为它修的是 prefill 的 cross-attention 缺失——一个结构规则(保住相邻 chunk 边界?)未必能覆盖的问题,这点值得实验。
- **驱逐 vs mask 的系统差异**:RA 强调 eviction 是唯一 bound 显存的路线(selection 不行);DA 强调 mask 可逆、不丢信息。若把 RA 的"保 prompt"推广为 DA 的"保 scaffold + 声明段",剩余部分在显存压力下用 per-head 随机驱逐兜底,可能是一条"mask 优先、随机驱逐兜底"的混合路线——既保留 DA 的可逆性/可读性,又获得 RA 的显存上界。

## 对本工作区的关联

- vLLM(`~/source_code/vllm/`)集成要点:压缩发生在 batched step 的同步点,content-dependent 打分在 paged 场景要多付一次 KV 扫描——评估任何 KV 压缩方案时这个 serving 放大效应必须计入(单次便宜 ≠ 服务便宜)。
- 与 Mooncake/LMCache 分层存储的关系:eviction 决定"哪些 KV 根本不存在",offloading 决定"存在的 KV 放哪层",两者正交;RA 的结论是驱逐层可以极简化。
- 对 agentic 长会话(如 kimi-code):prompt 保护规则的推广形式 = "保护系统提示 + 当前任务状态,随机/规则化驱逐旧 tool 输出",与 DA 的"tool 结果只对它被请求的那几步相关"判断一致。
