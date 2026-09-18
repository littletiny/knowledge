# Knowledge 库索引

本目录是论文/源码阅读知识库。`summary_*.md` 为单篇论文或项目的阅读总结,`discussion_*.md` 为围绕某个主题的讨论实录/综合笔记。**每次新增/更新/删除文档时,必须同步更新本索引并随同一 commit 提交。**

## 模型技术报告

| 文件 | 内容 |
|---|---|
| [summary_kimi_k3.md](summary_kimi_k3.md) | Kimi K3: Open Frontier Intelligence 技术报告 |
| [summary_dpskv4.md](summary_dpskv4.md) | DeepSeek-V4: 百万 token 上下文的高效智能 |
| [summary_dpskv32.md](summary_dpskv32.md) | DeepSeek-V3.2 技术报告 (arXiv 2512.02556) |
| [summary_glm5.md](summary_glm5.md) | GLM-5: from Vibe Coding to Agentic Engineering |

## Attention 与 KV Cache

| 文件 | 内容 |
|---|---|
| [summary_hisparse.md](summary_hisparse.md) | HiSparse: 层级化 KV cache 管理扩展 sparse-attention 解码 |
| [summary_hysparse.md](summary_hysparse.md) | HySparse: 混合 sparse attention + oracle token 选择 + KV 共享 |
| [summary_clsa_shared_routing.md](summary_clsa_shared_routing.md) | CLSA: 跨层 sparse attention 共享 routing |
| [summary_random_attention.md](summary_random_attention.md) | Random Attention: 重新思考推理场景的 KV cache 淘汰 |
| [summary_declarative_attention.md](summary_declarative_attention.md) | Declarative Attention: 让 LM 自主控制 attention |
| [summary_yoco.md](summary_yoco.md) | YOCO: Decoder-Decoder 架构,只缓存一次 |
| [summary_cacheblend.md](summary_cacheblend.md) | CacheBlend: RAG 场景的 KV cache 复用融合 |
| [summary_cache_dit.md](summary_cache_dit.md) | Cache-DiT: Diffusion Transformer 的 attention cache 机制分析 |
| [discussion_cache_dit.md](discussion_cache_dit.md) | Cache-DiT 讨论实录:从 KV cache 容量问题到 sglang 对照 |
| [discussion_kv_selection_signals.md](discussion_kv_selection_signals.md) | KV cache 优化三部曲讨论:选择信号从哪里来、值多少钱 |

## MoE / 推理系统 / Kernel

| 文件 | 内容 |
|---|---|
| [summary_deepgemm.md](summary_deepgemm.md) | DeepGEMM 源码阅读笔记(统一 tensor core kernel 库) |
| [summary_megamoe_megakernel.md](summary_megamoe_megakernel.md) | DeepGEMM MegaMoE megakernel 讨论笔记 |
| [summary_ktransformers_sosp25.md](summary_ktransformers_sosp25.md) | KTransformers (SOSP'25): CPU/GPU 混合 MoE 推理 |
| [summary_freetoken_edge_moe.md](summary_freetoken_edge_moe.md) | FreeToken: 边缘端带宽自适应 MoE serving |
| [discussion_consumer_gpu_moe_serving.md](discussion_consumer_gpu_moe_serving.md) | 消费级 GPU 跑前沿 MoE:SM89 FP8/sparse-MLA 问题与解法 |

## 分布式系统稳定性(Metastable Failures 研究线)

| 文件 | 内容 |
|---|---|
| [summary_metastable_synthesis.md](summary_metastable_synthesis.md) | 汇总入口:6 篇论文综合 |
| [summary_metastable_failures.md](summary_metastable_failures.md) | Metastable Failures in Distributed Systems (HotOS'21) |
| [summary_metastable_failures_wild.md](summary_metastable_failures_wild.md) | Metastable Failures in the Wild (OSDI'22) |
| [summary_analyzing_metastable.md](summary_analyzing_metastable.md) | Analyzing Metastable Failures (HotOS'25) |
| [summary_characterizing_metastable_faults.md](summary_characterizing_metastable_faults.md) | Characterizing Metastable Faults and Failures (arXiv 2026) |
| [summary_gray_failure.md](summary_gray_failure.md) | Gray Failure (HotOS'17) |
| [summary_breakwater.md](summary_breakwater.md) | Breakwater: μs 级 RPC 过载控制 (OSDI'20) |

## Agent / RL

| 文件 | 内容 |
|---|---|
| [summary_dream_rsi.md](summary_dream_rsi.md) | Dream-RSI: 通过演化世界实现递归自我提升 |
| [summary_wikiskill.md](summary_wikiskill.md) | WikiSkill: 把 agent 经验编译为持久知识 |

## 其他

| 文件 | 内容 |
|---|---|
| [summary_benchcad.md](summary_benchcad.md) | BenchCAD: 程序化 CAD 的工业级 benchmark |
| [summary_yoio_optical_flow.md](summary_yoio_optical_flow.md) | YOIO: 基于多重全局信息挖掘与融合的光流估计 |
