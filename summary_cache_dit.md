# Cache-DiT 缓存机制分析笔记

> 来源:对 `~/source_code/cache-dit/` 源码的逐层解读 + 以 MiniMax-H3(configs 见 `~/model-knowledge/configs/minimax-h3/`)为例的定量估算。
> 核心代码:`cache-dit/src/cache_dit/caching/`(`cache_contexts/`、`cache_blocks/pattern_base.py`、`calibrators/`)。

## 1. 核心结论

cache-dit(DBCache)**没有 LLM KV cache 那种容量问题**。它只缓存上一步的 2~3 个 block 级 residual 张量,覆盖式写入,显存恒定(O(1) 于步数),无分页、无 eviction/LRU——因为根本不需要。

## 2. 与 LLM KV cache 的本质区别

| | LLM KV cache | cache-dit |
|---|---|---|
| 推理方式 | 自回归,逐 token | 非自回归,每步全序列全量前向 |
| 缓存内容 | 历史 token 的 K/V,随长度线性增长 | 跨 denoising step 的特征 residual,固定大小 |
| 帧间依赖 | —— | 序列内全时空 attention(所有 latent 帧互相可见),不靠跨步缓存 |
| 步间状态 | KV 张量 | latent x_t 本身;K/V 每步重算、用完即弃 |
| 管理 | paged attention、块表、抢占/驱逐 | 字典覆盖写;形状变了直接判不命中 |

视频 DiT 的两个轴不要混淆:**帧间依赖 → 序列内 attention(横向);步间依赖 → latent 携带(纵向)**,两个方向都不需要 KV cache。
例外:流式/自回归视频扩散(CausVid、Self Forcing 等)用因果 attention + 帧级 KV cache + 滑动窗口,确实重新背上容量包袱。MiniMax-H3 不是这类:整段(≤15s)联合双向去噪。

## 3. DBCache 机制(Dual Block Cache)

把 N 个 transformer block 按**深度轴机械切三段**(无语义假设,对 token 一视同仁):

```
H(t,0) → [Fn 段: 前 n 块,每步必算] → H(t,n) → [Mn 段: 可跳过] → H(t,m) → [Bn 段: 后 n 块,每步必算] → 输出
              探针/指纹发生器            被加速主体(≈70% 算力)      精度校正器
下标:t = denoising step 序号,n = block 计数;token 在张量 seq 维内,不参与调度
```

- **指纹**:`R(t) = H(t,n) − H(t,0)`(Fn 段残差)。非线性来自 block 内部(softmax/LayerNorm/FFN);减 H(t,0) 只剥掉恒等直通支路,留下网络对内容的非线性响应。若 block 是线性的该方案退化成直接比输入。
- **判决**:`diff = mean|R(t)−R(t−1)| / mean|R(t−1)| < 0.08`(相对 L1,全部元素压成一个标量;`cache_manager.py:609-619`)。可选"显著 token 加权"(只在 diff 超阈值的 token 上统计)。
- **命中**:整个 Mn 段一个都不算(控制流跳过,非近似计算),`H(t,m) ≈ H(t,n) + Bn_residual`。Bn_residual 实为 Mn 段残差,命名意为"补到 Bn 之前的缺口"。
- **门卫**(`can_cache`,`cache_manager.py:883-914`):warmup_steps 开头强制全算、max_cached_steps、max_continuous_cached_steps(防连续命中漂移)、CFG 双分支独立缓存判决。
- **压缩的是每步计算量,不是步数**——调度器仍走满全部步。

## 4. 显存占用

- 缓存 = Fn 指纹(可 downsample)+ Mn 残差 + encoder 残差,开 separate CFG ×2 → **2~4 个全尺寸激活张量**,恒定,与步数无关(H3 768p 估算 ~2 GB)。
- `max_cached_steps` 是精度保护,不是容量限制。
- 例外:**DBPrune** 逐块缓存,buffer 随块数线性涨,故有 `force_reduce_calibrator_vram`。
- 校准器(缓存值的使用方式):复制(默认)/ TaylorSeer 外推(~2×order 张量)/ FoCa(4 个)/ DMD(history 窗口)。

## 5. DBPrune(逐块模式)

- logical block 区间宽度 w=1。判决**直接比 block 输入 hidden**(跨步 H_i(t) vs H_i(t−1)),不用残差(buffer 名 `{block_id}_Fn_original`,存的是原值);复用值是该块自己的残差(`{block_id}_Bn_residual`)。
- 为什么能直接比输入:逐块判决时,输入 H_i(t) 已吸收上游所有变化(含被跳块的近似),本身就是"流到这里的真实状态",零额外计算。
- 收益:全局 diff 大时可只重算变化剧烈的少数块;代价:判决开销 ×块数、显存随块数涨、控制流复杂。

## 6. 统一设计空间(三维权衡)

**指纹必须是"跑这段之前已经有的东西":**

| | TeaCache | DBCache | DBPrune |
|---|---|---|---|
| 区间宽度 | 整个 transformer | Mn 段 | 1 块 |
| 指纹 | timestep embedding(零成本) | Fn 残差(付 F 块计算) | 块输入(零成本) |
| 判据精度 | 最粗 | 中 | 细但碎片化 |

三个旋钮:**区间宽度 × 指纹成本 × 缓存值使用方式(复制/泰勒外推)**。
语义保真靠:Bn 校正段(每步总有真算部分)+ 连续命中上限 + warmup。

## 7. 为什么比残差而不直接比 hidden(H(t,8) vs H(t−1,8))

直接比 H 是合法配置(`is_l1_diff_enabled`,buffer 名 `Fn_hidden_states`),但默认不用:

- **信噪比**:ΔH = ΔH(0)(调度噪声漂移,共模) + ΔR(内容变化,信号)。早期 ΔH(0) 大 → H 模式漏判命中、白重算。
- **分母失真**:残差流累加使 |H| >> |R|,H 模式 diff 被系统性压小 → 阈值变松、不可跨模型迁移。
- 极端场景:latent 推进但内容静止时,R 模式正确命中,H 模式误重算。

## 8. 与扩散调度阶段的耦合 + 量化影响

- **早期步**:diff 天然大(结构形成期,内容剧变)→ 必然不命中 → 恰好该真算;warmup 双保险。缓存收益在中后期(纹理细化期)。
- **"两端惰性"设计哲学**:时间轴头部 warmup 守住、深度轴尾部 Bn 守住,压缩只发生在中段。注意两个"端"在不同轴上;去噪后期(时间轴尾部)反而是命中最多的地方。
- **残差小是残差网络的训练特性**(推理无梯度,与梯度爆炸无关),缓存方案白捡一个量程合适的探针。
- **量化**:量化误差是绝对量级 ε → R 模式 ε/|R| 大,diff 有误差地板、阈值附近抖动;H 模式数值更稳但语义更钝。FP8 接入需重调阈值。"值太大"的风险(超 FP8 量程/outlier)在 H 侧,"值太小"的风险(相对误差)在 R 侧——对 diff 判决,R 小是劣势不是优势。
- 隐藏红利:缓存残差意味着直通部分 H(t,8) 永远是当前步真值,缓存只提供"修正量",误差量级被 R 的小量程天然限制。

## 9. 定量参考:MiniMax-H3(24 FPS,≤15s,默认 768p)

压缩链路:视频 VAE 空间 16×16(`2,2,2,2`)、时间 4×(`1,2,2`);DiT patch [1,2,2] → **1 token ≈ 32×32 像素 × 4 帧**。主 DiT:50 层 + 2 refiner,hidden 5376,56 头 × 128(注意力内维 7168),FFN 14336,视频 24ch + 音频 32ch。

| 时长 | 输出帧 | latent 帧 | 视频 token(768p) |
|---|---|---|---|
| 5s | 120 | 30 | ~3 万 |
| 15s(上限) | 360 | 90 | ~9 万 |
| 30s(假设) | 720 | 180 | ~18 万 |

"KV cache"账(bf16,18 万 token):单层 K+V ≈ 5.2 GB;50 层全存理论值 ~258 GB——**瞬态激活而非持久缓存**,逐层复用后峰值仅单层 ~5-8 GB;注意力矩阵(单头 ~65 GB)由 FlashAttention 避免物化。手段:逐层调度 + FlashAttention + activation checkpointing + 序列并行。

H3 的 2K 方案(H3-Regenerate-2K)先 768p 再自重生成的动机:2K × 15s 直接生成 ~23 万 token 单次前向,成本不可行。

## 10. sglang diffusion 侧的对应实现(`~/source_code/sglang`)

sglang 的 diffusion 部分(`python/sglang/multimodal_gen/`,前身 FastVideo)缓存加速是三件套,block 级缓存**直接依赖 `cache-dit==1.3.0`**(`python/pyproject.toml:114`),判决逻辑零改动:

| | 来源 | 判决依据 | 粒度 | 缓存内容 | 显存 |
|---|---|---|---|---|---|
| cache-dit 集成 | 库依赖 | Fn 残差相对 L1(DBCache 原生) | Mn 段 | Fn 指纹 + Mn 残差 | 中 |
| TeaCache(自研) | `runtime/cache/teacache.py` | timestep 调制输入相对 L1 + 模型专属多项式 rescale,**累积**距离超阈值才算 | 整步 | 整步 residual,CFG 正负各一份 | 最省 |
| Spectrum(自研,实验) | `runtime/cache/spectrum.py` | 不判相似度;确定性 schedule(warmup 后每 window 步真算,间隔渐大) | 整步 | K 步完整 hidden 的 fp32 环形缓冲,Chebyshev 基岭回归外推 | 最大(大一个量级) |

sglang 在 cache-dit 之上的工程增量(`runtime/cache/cache_dit_integration.py`、`runtime/pipelines_core/stages/denoising.py:892`):

- 默认值按 serving 场景重调:`Fn=1, Bn=0, warmup=4, threshold=0.24, max_continuous=3`(比 cache-dit 默认 F8/0.08 激进,适配 Z-Image 等 few-step 蒸馏模型)。
- **并行一致性补丁**(`cache_dit_integration.py:55-118`):monkey-patch 相似度函数,对 mean_diff/mean_t1 在 TP×SP 组内 `all_reduce(AVG)`——serving 框架独有问题:各 rank 判决必须一致,否则集合通信挂死;单卡 cache-dit 无需考虑。
- 双 transformer(Wan2.2)适配、per-request 开关、与 breakable CUDA graph / FSDP 的互斥约束;与 DiT layerwise offload 兼容(跳过的块不加载)。
- 互补关系:cache-dit 管 block 级,TeaCache/Spectrum 管 timestep 级;few-step 蒸馏模型步数太少,block 级 warmup 开销占比高,整步跳过更划算。

sglang diffusion 的优化大头在别处:USP(Ulysses+Ring 混合序列并行)、十几种 attention 后端(FA3/4、SageAttention、VSA、VMoBA、Sliding-Tile 等,支持 per-component 覆盖)、NVFP4/SVDQ W4A4/GGUF 量化、breakable CUDA graph、encode/denoise/decode 三段分离调度、layerwise offload。支持模型覆盖 Wan2.1/2.2、FLUX.1/2、Qwen-Image、Z-Image、HunyuanVideo、SD3、LTX-2、MiniMax-H3 等约 30 个 DiT。

**关系结论**:核心机制同源同库;差异是"库 vs serving 框架"的关注点——cache-dit 管判决算法(指纹/阈值/校准器),sglang 管并行判决一致性、per-request 配置、与 graph/offload/量化共存。

## 11. 相关代码位置速查

cache-dit(`~/source_code/cache-dit/src/cache_dit/caching/`):

- 主流程(命中/未命中分支):`cache_blocks/pattern_base.py:220-327`
- DBPrune 逐块流程:`pattern_base.py:452-653`(`compute_or_prune`)
- similarity/L1 diff:`cache_contexts/cache_manager.py:552-626`
- can_cache 门卫:`cache_manager.py:883-914`
- 分段逻辑:`pattern_base.py:349-370`
- 校准器:`calibrators/taylorseer.py`、`foca.py`

sglang(`~/source_code/sglang/python/sglang/multimodal_gen/`):

- cache-dit 集成与并行补丁:`runtime/cache/cache_dit_integration.py`
- TeaCache:`runtime/cache/teacache.py`;Spectrum:`runtime/cache/spectrum.py`
- 去噪循环与缓存挂载:`runtime/pipelines_core/stages/denoising.py:892`(`_maybe_enable_cache_dit`)

其他:

- H3 config:`~/model-knowledge/configs/minimax-h3/`(transformer/vae/audio_vae/text_encoder + FL2VA/Ref2VA 变体)
- 完整对话记录:`~/knowledge/discussion_cache_dit.md`
