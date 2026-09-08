# KV Cache 优化三部曲讨论:选择信号从哪里来、值多少钱

- **日期**: 2026-09-08
- **涉及论文**(详细总结见各自文件):
  - CacheBlend(arXiv 2405.16444)→ `summary_cacheblend.md`
  - Declarative Attention / DA(arXiv 2609.02737)→ `summary_declarative_attention.md`
  - Random Attention / RA(arXiv 2609.03430)→ `summary_random_attention.md`
- 本文档记录围绕这三篇论文展开的横向讨论:"attention 该看谁"的决定可以从哪里获得,以及 RA 与 SWA、DeepSeek-V4 压缩、DSA indexer 之间的同构与本质差异。

## 1. 三篇论文的定位:选择信号的三种来源

三篇论文处理同一个根本矛盾:**长上下文/长推理下,模型实际需要的信息只是很小一部分,但默认实现要为全部上下文付全价**。它们恰好覆盖推理生命周期的三个阶段,且代表选择信号的三种来源:

| | CacheBlend (2405.16444) | DA (2609.02737) | Random Attention (2609.03430) |
|---|---|---|---|
| 阶段 | prefill(跨请求 KV 复用) | decode(每步读哪些 KV) | decode(哪些 KV 永久丢弃) |
| 选择信号 | **extrinsic**:实测 KV 偏差(HKVD) | **intrinsic**:模型在 CoT 里文本声明 | **无**:结构规则(保 prompt + 每 head 随机) |
| 粒度 | token 级、逐层 | segment 级(~2K magic chunk) | token 级、per KV head |
| 可逆性 | 重算修复(15% token) | mask 可逆、不驱逐 | eviction 不可逆、但 bound 显存 |
| 核心结论 | 15% 重算即可恢复 cross-attention,TTFT ↓2.2–3.3× | 声明式 mask 省 31–52% attended tokens,准确率掉 1–3pp | 排序信号几乎不贡献准确率,吞吐 +32–43% |

互补性:CacheBlend 省 prefill,DA 省 decode 带宽,RA 给显存硬上界——理论上可以拼成一条完整流水线(chunk KV 离线预计算 → DA prompt 组装 → decode 声明式 mask → 显存压力下随机驱逐兜底)。

## 2. RA 是不是 SWA 的变种?——一半是,论文自己承认了

RA 的存活结构 = **prompt 钉死 + 软 recency(存活概率 ~0.94ⁿ/轮)+ 每 head 不同的稀薄长尾**。RA 附录(`sec:keeplog`)明确称其 "a soft recency window"。更直接的证据:机制实验里的 **recency+prompt 对照组**(保 prompt + 连续最近窗口,即 StreamingLLM 式)在真实任务上与最强 baseline 只差 2 分以内——主干收益确实来自"保 prompt + 近期窗口"这个结构,而非随机性本身。

但 RA ≠ 硬 SWA,两个实质差异:

1. **旧位置存活概率非零 vs 零**:硬窗口外信息必然丢失;RA 长尾让每个旧位置在每个 head 都有非零存活率。keep-log 实测:1–2k token 年龄带,RA 跨 head 并集覆盖率 0.776,shared-draw(所有 head 同样位置)只有 0.161。
2. **跨 head 多样性**:真实轨迹上不重要(文本级复述冗余已够,shared-draw 只差 0.3 分),但对"只说一次的事实"是生死线(planted-fact probe:1 head 检索率 3% → 8 head 99%,超加性)。

**结论:RA = SWA(保 prompt 版)主干 + 随机长尾提供的稀疏全历史覆盖。** RA 的价值是把这个混合物立为 null hypothesis:任何选择信号必须在同预算、同 prompt 保护下打败它,否则证明没有从信号中提取出可用信息。

## 3. DSV4 压缩 + SWA 与 RA 的同构:骨架相同,保真度维度相反

两者都是 **"近期全保真 + 远期低保真"** 的双层访问模式:

| | 近期 | 远期 |
|---|---|---|
| RA | 软 recency(近期几乎全在) | **逐位置随机采样**:verbatim 但稀疏(大部分位置丢,留下的无损) |
| DeepSeek-V4 | SWA 层 + indexer top-k 全分辨率读取 | **逐位置有损压缩**:128:1 compressed MQA,稠密覆盖但每位置只剩摘要 |

低保真的实现方式恰好相反:RA 是 **sparse sampling of lossless tokens**;DSV4 是 **lossy compression of all positions**。由此产生两个本质差距:

- **召回能力**:DSV4 的 indexer 做 query 依赖的 top-k 召回,能把远期全分辨率条目捞回;RA 没有任何召回机制,needle 只能靠运气留在长尾(passcode probe:RA 检索率 0,R-KV 84%)。
- **显存语义**:RA 是 eviction,显存硬上界;DSV4 是训练进架构的压缩存储 + selection,存储仍随 context 线性增长(只是常数小 1–2 个数量级;DA 论文附录 KV 字节表:DSV4-Flash 3.5 KB/token stored,448 B/token per-step read)。

另外 DSV4 是训练进去的(模型学会在该访问模式下工作),RA 是 training-free(白嫖现有冗余)。"类似"止步于宏观形状。

## 4. Indexer 与 RA 的 top-k:机械同构,信息论对立

表面上都是"每位置一个标量分数,取 top-K",但信息来源是两个极端:

```
RA:      s_i = Uniform(0,1)             ← 零信息,query-independent,每 64 步一次,eviction(不可逆)
Indexer: s_i = f_learned(q_t, k_i)      ← 内容+query 依赖,每步 O(N) 一次,selection(可逆)
DA:      从模型生成文本 O(1) 解析        ← 声明式,span 粒度,mask(可逆)
```

Indexer 的分数编码"当前 query 需要什么";RA 的分数什么都不编码。RA 论文的意义:在推理轨迹上,前者相对后者测不出增量(同预算同保护下)。边界同样清楚:一旦任务是"单次陈述、不复述、很久后用"(RULER NIAH 主战场,也是 DA 的评测场景),零信息分数必死,内容依赖信号才有存在意义。

## 5. 收拢:骨架趋同,分歧只在"远期层放什么、谁决定"

三种路线的**访问模式骨架趋同**:近期全保真 + 远期低成本 + 结构性强保护(prompt/scaffold)。分歧只在远期层的实现:

- **RA**:放随机 verbatim 样本,无人决定(null hypothesis 的下限)
- **Indexer/DSV4**:放压缩稠密覆盖 + 学习到的 query 依赖召回(训练上限)
- **DA**:放模型自己声明要看的段(用语言能力替代学习打分器)

RA 论文证明了这个骨架(保护 + 覆盖)解释绝大部分准确率;骨架之上的选择机制只在 needle regime 能证明增量。DA 论文则从另一端证明:选择可以零成本地从模型自己的文本里读出来,且天然可审计。

## 6. 深层呼应与混合路线设想

- **RA 解释了 DA 为什么敢"不看"**:RA 发现推理轨迹靠"复述"获得文本级冗余;DA 的 `<focus>` 模式制度化了复述(verbatim 抽取值到 response),`<local>` 模式正是"只看复述出的工作区"。DA 主动制造了让随机驱逐可行的冗余结构。
- **regime 互补**:RA 针对"短 prompt + 长生成"(prompt 脆弱);DA 针对"长 context + 推理"(context 巨大、scaffold 常看)。RA 的失效区(单次陈述 needle)正是 DA 的主场。
- **对打分派的共同批判**:RA 说 decode 驱逐的打分不增值;DA 说每步 O(N) 代理扫描可换成 O(1) 文本声明。共同指向:选择信号应从结构/文本免费获得,而非每步计算。CacheBlend 的 HKVD 打分能成立,是因为它修的是 prefill 的 cross-attention 缺失——一个纯结构规则未必能覆盖的问题,值得实验验证。
- **可能的混合路线**:DA 的"保 scaffold + 声明段"作为 mask 优先层(可逆、可读),RA 的 per-head 随机驱逐作为显存兜底层(硬上界),indexer/压缩作为 needle 保险(训练后)。三层各取所长。
- **serving 放大效应**(RA 效率节的教训,适用于所有 KV 压缩方案评估):单次打分便宜 ≠ 服务便宜——128 并发下压缩发生在 batched step 同步点、全体等待;fused kernel 不产出 attention 统计,content-dependent 打分需额外 KV 扫描。

## 7. 对本工作区的落点

- **vLLM**:DA(hook metadata builder 改 block table)和 RA(per-head 物理驱逐 + compaction)都是不碰 kernel 的低成本集成路径,可互相参考;评估任何方案时必须计入 serving 放大效应。
- **Mooncake / LMCache**:eviction 决定"哪些 KV 不存在",offloading 决定"存在的 KV 放哪层",DA 的声明提供"何时预取"的提前信号——三层正交可组合。DA 的"可逆 context compaction"设想的加载端可复用 CacheBlend/LMCache 的流水线加载。
- **agentic 长会话(kimi-code 类)**:RA 的 prompt 保护推广为"保系统提示 + 当前任务状态,规则化驱逐旧 tool 输出";DA 的判断"tool 结果只对它被请求的那几步相关"与之方向一致;tool 结果天然可寻址,是 magic chunk 的真实形态。
