# Declarative Attention: Language Models Can Control Their Own Attention

- **arXiv**: 2609.02737 (2026-09, KAIST AI + Google DeepMind; Namgyu Ho, Huzama Ahmad, Woosung Koh, Se-Young Yun, Tal Schuster, Cicero Nogueira dos Santos)
- **TeX 源码缓存**: `~/.cache/nanochat/knowledge/2609.02737/`

## 一句话总结

让 LLM 在 CoT 里**自己用文本声明"下一步我要看哪段上下文"**(`<global>` / `<focus magic_chunks="K">` / `<local>` 三种模式),推理引擎像解析 tool call 一样解析这些声明,据此动态改写 KV cache block table 做 attention mask —— 零样本、免训练,在 15 个长上下文任务上把 decode 阶段的 attended tokens 砍掉 52%(Gemma-4-31B)/ 31%(Qwen-3.6-27B),准确率只掉 1.27pp / 2.75pp。

## 问题动机

- 长上下文 decode 是 memory-bound 的:每个 decode step 都要把整个 KV cache 从 HBM 读一遍。例:Qwen-3.5-397B-A17B 在 1M context 下每步要读 ~15 GB KV cache,与加载 17B 激活参数量相当。
- 已有稀疏注意力都是 **extrinsic** 的:用代理分数(page min-max key bounds 的 Quest、learned indexer 的 DSA/NSA 等)预测哪些 token 重要,**每步仍是 O(N) 扫描**(只是常数小)。
- 本文提 **intrinsic** 路线:模型自己"知道"该看哪——hidden states 已编码未来 token 信息,CoT 把它外化成文本。把 CoT 从"想**什么**"扩展到"看**哪里**"。mask 直接从生成的文本读出,**选择成本为 O(1)**(解析一个 tag),而不是每步 O(N)。

## 方法:Declarative Attention (DA) 协议

### 三种 attention 模式

| 模式 | 可见内容 | 用途 |
|---|---|---|
| `<global>`(默认) | 全部 context 段 | 导航:浏览全上下文,定位下一个要 focus 的段 |
| `<focus magic_chunks="K">` | 只有第 K 段(可加多个) | 从指定段里 verbatim 抽取事实 |
| `<local>` | 不看任何 context 段 | 基于已抽取的内容做规划/综合答案 |

scaffold(system 指令、question、DA 指令)在**所有模式下始终可见**;模式切换以 closing `>` 为界,解析规则极简。`<global>` tag 本身只是提示结构(默认即 global),不产生状态切换。

### Context 投递方式:magic chunks + 模拟 tool-use transcript

- 把长 context 切成 **~2048 token 的 "magic chunk"**(目标 2048,硬上限 2560),切分按分隔符层级:段落(双换行)→ 单换行 → 句末 → 子句 → 词边界,**绝不在词中间切**;无空白的 base64 类长串作为原子单元不拆。工作在字符偏移空间,保证无损 partition。
- 关键设计:段边界**对齐模型训练时见过的边界** —— 用模型原生 tool-call 格式把 context 伪装成"已经逐段检索完毕的 tool-use 对话"(声明 `get_magic_chunk` 工具,每个 assistant turn 后跟一个 `Magic Chunk N` 的 tool response),实际上工具从未执行,所有段在生成前就全部就位。命名 "magic chunk" 是为了避免模型把它当成文档自带的章节。
- 动机:模型对 user/assistant/tool turn 的特殊 token 边界(后训练全程见过)的跟踪远比任意新造分隔符可靠。

### 推理引擎集成(vLLM)

- **DA state machine** 挂在 vLLM 的 attention metadata builder 上,**不改 kernel、不改 scheduler**:每个 decode step 解析输出流中的 tag,然后**原地改写该 request 的 KV cache block table**,只保留要看的 block,FlashAttention 等现有 kernel 原样跑,只是读得少。
- **Block 对齐**:mask 向外取整到 KV block 边界(vLLM block 一般 16–32 token),每条保留 span 每侧最多多出一个 block —— 相对 2048-token 段可忽略。只有整 block 跳过才能真正省内存读取(零散的 token 级 mask 不省带宽)。这正是 NSA 的 block-sparse 原则。
- **只作用于 global attention 层**:SWA(Gemma-4,1024 窗口)和 GDN(Qwen-3.5/3.6,线性注意力)层 per-step 成本与 context 长度无关,mask 无利可图,不碰。
- 始终保留三个区域:前 16 token 的 attention sink(放一个固定短 system 指令占据)、question+指令到 prompt 末尾的 local window、以及已生成的 response 全文。

## 效率模型:roofline wall-time

- decode 拆成三项:**matmul**(compute-bound,~2P FLOPs/step,MFU 40%)、**global memory**(global attention 的 KV 读,memory-bound,MBU 70%)、**local memory**(SWA/GDN 的固定 per-step 读,DA 不碰)。
- 量级对比(1M context, B200):attention ≈ 2.7 ms/step vs FFN ≈ 0.019 ms/step,差 **~145 倍**;KV 读与激活参数加载量相当,但参数加载可以跨 batch 摊销、KV 读不能。
- 结论:DA 的收益在**大 batch、长 context、PD 分离**的 serving 场景最大 —— 此时 attention 是 decode 的主导成本(vanilla 下 Gemma 73%、Qwen 86% 的 decode 时间花在 global KV 读上)。

## 实验结果

- **模型**:Gemma-4-{31B, 12B, E4B} + Qwen-3.6-27B / Qwen-3.5-{9B, 4B},B200 + vLLM,非 thinking 模式,8K 生成上限。
- **评测**:15 个长上下文来源(RULER、LongBench v1/v2、LooGLE、ZeroSCROLLS),单 span 检索 + 多 span 推理两类,context 最长 244K;LLM judge(Qwen-3.5-4B + Gemini-3-Flash 生成 rubric,与 Gemini-3.1-Pro 的 Pearson r=0.99,单条一致率 98.53%)。
- **主结果**:DA 平均准确率 87.01→85.74(Gemma)/ 85.31→82.56(Qwen);attended tokens 13.43M→6.45M(−52.0%)/ 22.54M→15.52M(−31.1%)。
- **消融**:DA-no-mask(同样 prompt 但全 attention)准确率几乎无损(分块 prompt 格式本身免费),但 token 数反而比 vanilla 高 66%/29%(协议导致 decode step 多 ~31–35%);**mask 把这部分开销扭转为净节省** —— 相对 DA-nm 砍掉 71.1%/46.5%。所以效率全部来自 mask,而准确率损失也主要来自 mask。
- **规模 scaling**:相对准确率从 Gemma-4-E4B 的 29% 单调升到 31B 的 99%;per-step 注意力比例约 0.5、基本与模型规模无关;E4B 崩溃主要是协议遵循失败(focus 解析成功率仅 58%,大模型 99%)。
- **Context scaling**:32K 以内准确率贴住 vanilla,最长 bin 仍有 ~96%;绝对节省随 context 增长(最长 bin 省 ~21M tokens/response),per-step 节省比例 ~50% 恒定。
- **模式统计**:focus+local 占生成 token ~73%,per-token 省 76–99%;但 global 模式仍占 attended tokens 的 80%+,且随 context 变长占比上升(最长 bin ~45–55%)。
- **Wall-clock 估计**:B200 上 decode 时间降到 vanilla 的 0.71×(Gemma)/ 0.77×(Qwen);Gemma 的 SWA 层(50/60 层)构成 42% 的 attention 时间下限,稀释了节省,而 Qwen 的 GDN state 只占 5%。
- **失败案例**(附录,6 个来源):① 证据被切分破坏(跨段全局计数 cwe、被切开的表格 structured_data)→ 应做结构感知切分 + map-reduce;② 输出长度随文档增长(逐段摘要、全文排序等)→ decode step 与文档等长,逐 token 节省被步数抵消 → 应把长输出也路由进 focus。

## 讨论与展望(最有价值的部分)

- **System-2 稀疏注意力**:稀疏选择以语言表达,可纯靠改指令改变策略、随模型规模自动变强、天然可审计(驱动 KV 读的 token 就是人能读的文本);未来的 RLVR 可以同时奖励准确率和 attention 效率。
- **与现有技术叠加**:global 模式可以叠加 Quest/DSA 式轻量扫描;speculative decoding 与 DA 互补(一个降每步成本,一个合并步数,mode 内 mask 固定所以 drafting 不受影响)。
- **System-2 KV offloading / 可逆 context compaction**:DA 的 mask 只在 span 边界变化且**提前在文本中宣告**,天然适合做 KV cache 分层:失焦段 spill 到 host memory,被 focus 声明点名时预取回。与 Anthropic/OpenAI 的 compaction(丢弃原文 KV、召回要 re-prefill)不同,DA 序列从不编辑,off-device 的 cache 保持有效 —— 可逆压缩、免 re-prefill。这与 Mooncake(`~/source_code/mooncake/`)的 KV 分层方向直接相关。
- **局限**:zero-shot 策略次优(步数多 1/3)、benchmark 需人工切段(agentic 场景天然有 tool call/user turn 边界)、thinking 模式下模型不遵守协议(可作为工具声明暴露给 thinking trace)、global 模式仍占大头(未来可用 in-context 段索引替代全保真浏览)。

## 附录亮点:当前架构的 KV 字节横评(截至 2026-08)

论文给了一张非常有用的 **per-context-token KV 字节数表**(stored vs O(N) per-step read),与 `~/model-knowledge/configs/` 里的模型直接对应:

| 模型 | 机制 | stored B/tok(as served) | O(N) read B/tok | 1M ctx attention 占 decode 时间 |
|---|---|---|---|---|
| MiniMax-M2.7 | 全 attention GQA 62层 | 253,952 | 同 | (满 attention 档) |
| Qwen3.8-Max | 全 attention 23层+GDN | 47,104 (FP8) | 同 | 98.15% |
| Kimi-K3 | 全 attention MLA 24层+KDA | 13,824 (FP8) | 同 | 94.11% |
| MiniMax-M3 | 57 MSA + 3 全 attention | 137,472 | 20,736 | 97.50% |
| GLM-5.2/5.3 | top-2048 sparse MLA,21 全 indexer | 53,940 (FP8) | 2,772 | — |
| Qwen3.8-Flash-Next | QSA top-2048,每 4 token 一个压缩 index key | 25,344 | 768 | — |
| GLM-5.3-Flash | sparse MLA,pooled indexer | 7,040 (FP8) | 352 | 56.49% |
| DeepSeek-V4-Pro | 压缩 MQA,30 层 4:1 + 31 层 128:1 | 5,024 | 651 | 66.55% |
| DeepSeek-V4-Flash | 同上,top-512 | 3,509 | 448 | 83.22% |

要点:1M context 下所有模型的 attention 仍占 decode 的 56–99%;但 128K 时 indexer 系模型有三行跌破 50% —— **context 长度对 attention 占比的影响远大于 MFU/MBU 假设**。indexer 系的 stored 与 O(N) read 相差 6.6–33×,top-k 主注意力读取不随 context 增长,归入 local 项。

## 与 `~/source_code/` 的关联与启发

- **vLLM 集成模式值得参考**:DA 不改 kernel/scheduler,只 hook attention metadata builder 改写 block table(`vllm/` 的 block table 机制)。这是做"引擎侧自定义稀疏"的低成本路径;同类手法在 ThinKV 里也出现过(同一个 block table 数据结构,用于复用被驱逐的 slot)。
- **与 DSA(`summary_dpskv32.md`、`summary_dpskv4.md`)的关系**:正交互补。DSA 每步用 learned indexer 做 O(N) 轻量扫描;DA 在 focus/local 步把扫描降到 O(1),但 global 步仍需全量 —— 论文建议 global 步叠加 DSA 式扫描。GLM-5.2/5.3、DeepSeek-V4 的 indexer 摊销(IndexShare/IndexPool)说明工业界在压 O(N) 常数,DA 压的是"多少步需要 O(N)"。
- **对 agentic 长上下文(如 kimi-code 类 CLI agent)最直接**:tool call 结果天然就是可寻址的段,且"tool 结果只对它被请求的那几步相关,却要留在 context 里陪跑全程"正是 DA 省钱的场景。可以把 DA 模式想成给 agent runtime 加一层"attention 作用域提示"。
- **KDA/GDN 混合架构下的启示**:Gemma-4 的 SWA 层稀释了 42% 的节省;对 Kimi-K3 / GLM-5.3-Flash 这类线性注意力占大头的模型,DA 能动的只剩少数 global 层 —— 但这些层恰恰是 1M context 下占 94% decode 时间的部分(K3),所以 DA 仍有价值,且与 indexer 方案正交。
- **可逆 context compaction + KV offload(Mooncake)**:mask 变化在 span 边界、提前宣告,是唯一支持"预取"的稀疏模式;比 MemGPT 式驱逐-召回(召回要 round trip/re-prefill)便宜。
