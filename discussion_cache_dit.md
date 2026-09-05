# Cache-DiT 讨论实录:从 KV cache 容量问题到 sglang 对照

> 一次完整的技术讨论记录(Q&A 形式),围绕 `~/source_code/cache-dit/` 的缓存机制展开,逐步深入到 MiniMax-H3 的定量估算、DBCache/DBPrune 设计细节、量化影响,最后对照 sglang 的 diffusion 实现。
> 蒸馏版总结见 `~/knowledge/summary_cache_dit.md`;H3 configs 在 `~/model-knowledge/configs/minimax-h3/`。

---

## Q1: cache-dit 的缓存实现是什么?有类似 KV cache 的容量问题吗?

**A: 没有容量问题。** 核心机制是 DBCache(Dual Block Cache):

- **缓存粒度是 transformer block 组,不是 attention K/V**。把 N 个 block 按深度切成三段(`cache_blocks/pattern_base.py:349`):
  - Fn(前 n 块,默认 F8):每步必算,输出残差 `R(t) = Fn输出 − Fn输入` 当"指纹"
  - Mn(中间块):被缓存/跳过的主体,完整计算时缓存其残差
  - Bn(后 n 块,默认 B0):每步必算,做精度校正
- **判决**(`cache_contexts/cache_manager.py:552`):当前步与上一步 Fn 残差的相对 L1 diff `mean|R(t)−R(t−1)| / mean|R(t−1)| < 0.08` 则命中,跳过整个 Mn 段,用缓存残差加回。
- **存储就是普通字典**(`cache_context.py:36` 的 `buffers: Dict[str, Tensor]`),同名 key 覆盖写,只留上一步的 2~4 个全尺寸激活张量 `[B, seq_len, hidden_dim]`(开 CFG 分离 ×2;FLUX 级几十~几百 MB)。
- 没有历史堆积、没有容量上限、**没有 eviction/LRU**。`max_cached_steps`/`max_continuous_cached_steps` 是精度保护,不是内存限制。

**为什么和 KV cache 不同**:DiT 推理非自回归,每个 denoising step 对完整序列全量前向,K/V 当步重算、用完即弃,不存在"历史累积"这个根因。缓存的是跨 step 的特征复用(step 级),不是 token 级历史。

扩展机制(TaylorSeer ~2×order 张量、FoCa 4 个、DMD history 窗口)都是固定小常数倍,仍 O(1) 于步数。唯一显存随模型增长的例外是 **DBPrune**(逐块缓存),故代码里有 `force_reduce_calibrator_vram`。

## Q2: 这类视频生成模型,一帧有多少 token?

**A:** 路径是 `像素帧 → 3D VAE(空间 8×8、时间 4× 压缩)→ latent → patchify(通常 1×2×2)`,所以**每 latent 帧 token ≈ (H/16)×(W/16)**;注意 4 个输出帧才对应 1 个 latent 帧。

| 模型/配置 | 每 latent 帧 | latent 帧数 | 总序列 |
|---|---|---|---|
| Wan2.1 480×832, 81帧 | ~1,560 | 21 | ~32,760 |
| Wan2.1 720×1280, 81帧 | ~3,600 | 21 | ~75,600 |
| CogVideoX 480×720, 49帧 | ~1,350 | 13 | ~17,550 |
| HunyuanVideo 720×1280, 129帧 | ~3,600 | 33 | ~118,800 |
| LTX-Video(特例,空间 32×/时间 8×) | ~384 | — | 几千 |

摊到每个输出帧约 300~900 token;总序列 3万~12万,是图像 DiT(FLUX 1024² 约 4k)的 10~30 倍——视频里 attention O(n²) 成为绝对瓶颈,这正是跨步缓存首先拿视频开刀的原因。

## Q3: (任务)下载 MiniMax-H3 作为参照

按 `~/AGENTS.md` 约定放入 `~/model-knowledge/configs/minimax-h3/`(经 hf-mirror,本机直连 huggingface.co 不通)。**只存 config.json,无权重**——完整快照约 498.5 GB,磁盘仅剩 ~12 GB 也放不下。

- 不是单模型仓库,是 diffusers 风格流水线:transformer(主 DiT)+ transformer_ref + vae(视频)+ audio_vae + text_encoder,外加 FL2VA/Ref2VA 两个完整变体,共 15 个 config.json(132K)。
- 主 DiT 架构:50 层 + 2 refiner 层,hidden 5376,56 头 × 128 dim(注意力内维 7168 > hidden),FFN 14336,patch [1,2,2],视频 24ch + 音频 32ch。
- README 关键参数:**24 FPS**,输出时长 4–15 秒,默认短边 768p,2K 靠 H3-Regenerate-2K 重生成。
- `~/AGENTS.md` 的 configs 目录树已补 minimax-h3 条目。

## Q4: 30s 视频,每秒多少帧?token 是不是不少?

**A:** H3 是 **24 FPS**,30s = 720 帧(注意 H3 原生上限 15s,30s 是假设场景)。

H3 压缩更狠:VAE 空间 **16×16**(`2,2,2,2`)、时间 4×(`1,2,2`),patch [1,2,2] → **1 token ≈ 32×32 像素 × 4 帧**:

| 时长 | 输出帧 | latent 帧 | 视频 token(768p,每 latent 帧 ~1000) |
|---|---|---|---|
| 5s | 120 | 30 | ~3 万 |
| 15s(上限) | 360 | 90 | ~9 万 |
| 30s(假设) | 720 | 180 | ~18 万 |
| 30s + 2K | 720 | 180 | ~45 万 |

token 确实多,但 H3 已算省的(16× 空间压缩 vs Wan 的 8×,同规格只有 Wan 的 ~1/4)。另有音频 token(audio 32ch)拼进同一个 50 层 DiT。2K 不直生的动机:2K × 15s ≈ 23 万 token 单次前向,成本扛不住。

## Q5: 算下 KV cache,看起来也不少?

**A:** 绝对量确实大,但性质完全不同。H3 参数:50 层,注意力内维 7168,bf16,18 万 token:

```
单层 K+V: 180,000 × 7,168 × 2B × 2 ≈ 5.2 GB
50 层全量 K+V(理论值):           ≈ 258 GB
```

- LLM KV cache 是**持久**的:跨自回归步累积、常驻显存 → 分页/驱逐/容量规划。
- DiT 的 K/V 是**瞬态激活**:每步全量重算,用完即弃;前向逐层推进,任何时刻只有当前层 Q/K/V 是活的,峰值 ~5–8 GB。
- 注意力矩阵本身(18万² × 56 头,单头 ~65 GB)从不物化——FlashAttention 的意义。
- 兜底手段:activation checkpointing、Ulysses/Ring 序列并行、cache-dit 式跨步缓存。

逻辑链:token 多 → FlashAttention 让矩阵不物化 → K/V 每层 5GB 瞬态逐层复用 → 真正成本是"50 步 × 每步全量重算" → cache-dit 用几百 MB 固定缓存砍掉一半以上计算。

## Q6: DiT 不用存历史值?视频不是要处理连续帧依赖吗?

**A: 帧间依赖在"一次前向内部"解决,不靠存历史。** H3/Wan 是 joint denoising:整段视频所有 latent 帧拼成一条序列(90 帧 × 1000 token),一次前向内做**双向全时空 attention**——"历史帧"不在过去,就在当前序列里,帧与帧直接互相可见。

扩散的时间轴是**噪声轴**不是帧轴:步与步之间传递的状态是 latent x_t 本身(整段视频的噪声版本),K/V 从新的 x_t 重新投影——x_t 变了旧 K/V 本就失效。

**例外**:流式/自回归视频扩散(CausVid、Self Forcing 等)用因果 attention、逐 chunk 生成,**真的维护帧级 KV cache + 滑动窗口**——回到 LLM 那套容量问题,换取无限时长/低延迟,代价是长程一致性退化。H3 联合生成方向相反:一致性好但时长被卡在 15s。这是两种范式的分野。

## Q7: 所以 cache-dit 只需存最近 step 求 delta 做 step 压缩?

**A: 本质正确,两处修正:**

1. 压缩的是**每步计算量**而非步数——调度器仍走满 50 步(步数决定噪声轨迹),只是约半数步变成"Fn + 加法 + Bn"的廉价步。叫 step 级计算跳过更准确。
2. 复用的是**变化量(residual)而非特征本身**,Bn 段得以在新输入基础上校正——这是它比 DeepCache/FasterDiffusion 稳的原因之一。

能做这件事的前提:DiT 无跨步状态依赖,相邻步特征高度相似;跳错了误差也不会自回归式累积(后续步仍在去噪,轨迹自带纠错)。TaylorSeer 把"直接抄上一步"升级为"泰勒外推预测",存的仍是固定几个历史 residual。

## Q8: 展示计算图,L1 distance 是针对哪个流程的结果?

**A**(`pattern_base.py:220-327` 实际代码):

```
H(t,0) → [Fn 段,每步必算] → H(t,n)
                              │
              指纹 R(t) = H(t,n) − H(t,0)   ← L1 diff 对这个残差算 ★
                              │
        diff = mean|R(t)−R(t−1)| / mean|R(t−1)| < 0.08 ?
            ┌────命中────┐          ┌────未命中────┐
   H = H(t,n) + 缓存残差     Mn 段完整计算
   (Mn 整段跳过)            写入: R(t)→Fn buffer, Mn残差→Bn buffer
            └──────┬───────┘
              [Bn 段,每步必算,当前特征上校正] → 输出
```

细节:
- **不直接比输入**是因为输入 latent 每步被调度器注入新噪声(共模漂移),Fn 段充当特征提取器("for more stable diff calculation",`pattern_base.py:224`)。
- 比的是**相对差**(除以 mean|R(t−1)|),阈值才有跨模型可迁移性;可选显著 token 加权(`cache_manager.py:591-606`)。
- 判决只用 Fn 残差一个张量;被复用的是 Mn 段残差(buffer 名 `Bn_residual`),encoder 有平行缓存但不参与判决。
- 备选模式 `is_l1_diff_enabled`:直接对 Fn 输出 hidden 比 L1(buffer 名 `Fn_hidden_states`)。
- CFG 双分支默认各自缓存判决;`cfg_diff_compute_separate=False` 时复用非 CFG 分支的 diff。

## Q9: 解释 Fn / Mn / Bn 等变量

**A:** F/M/B = First / Middle / Back,n = 块数,是"段 + 块数"的缩写,按 block index 的连续区间切:

```
H(t,0) → [Fn 前n块] → H(t,n) → [Mn 中间块] → H(t,m) → [Bn 后n块] → H(t,N)
          探针/指纹            可跳过/缓存            精度校正
```

以 FLUX ≈27 块 F8B0 为例:Fn = block 0~7,Mn = block 8~26,Bn = 空。

**易混淆点**:`Bn_residual` buffer 存的其实是 **Mn 段的残差** `H(t,m) − H(t,n)`;命名逻辑是"Bn 段之前的一切"——命中时 `H(t,m) ≈ H(t,n) + Bn_residual`,一步从 Fn 输出跳到 Bn 输入。段名描述执行顺序,buffer 名描述"补上到 Bn 之前的缺口"。

## Q10: Fn/Mn/Bn 是按语义切的吗?还是 token 粒度?

**A: 都不是,是按网络深度(block index)机械切的连续区间**,无语义假设,对 token 一视同仁——同一 step 所有 token 走相同路径,判决是整个 step 一个标量,命中则整条序列一起跳。

- 选择依据是纯工程启发:前几块对输入变化最敏感(适合当探针)、中间块特征演化最平缓(跳过损失最小)、末尾块决定输出(留着校正)。
- 唯一沾 token 边的是判决统计的显著 token 加权——只影响判据怎么算,不影响跳过粒度。
- token 级缓存(静止区域复用、运动区域重算)是另一条技术线;cache-dit 不走是因为 gather/scatter + 变长 attention 的 kernel 开销常吃掉收益,而 block 级跳过对 FlashAttention/编译/序列并行完全透明。

## Q11: block 之间靠 hidden 衔接,L1 distance 怎么处理的?H(t,n) 的计算流程?

**A: H(t,n) 不是专门算的量,就是第 n 个 block 的输出 hidden,算完顺手存引用。** 第 t 步:

```
H(t,0) = patch_embed(x_t) + 条件     ← [B, seq_len, D]
H(t,i) = H(t,i-1) + block_{i-1}(H(t,i-1))   ← 每块自带残差连接,形状不变
...
H(t,n) ← 第 n 块的输出
```

L1 计算(`cache_manager.py:609-619`,三行):

```
R(t)       = H(t,n) − H(t,0)            逐元素,形状不变
mean_diff  = mean(|R(t) − R(t−1)|)      全部元素平均 → 标量
diff       = mean_diff / mean(|R(t−1)|) 归一化,与 0.08 比较
```

"L1" 就是逐元素绝对差再平均,不是向量距离。

**为什么比残差不比 H(t,n)**:每个 block 输出 = 输入 + 增量,H(t,n) 里含"每步必变的输入直通";减去 H(t,0) 消掉共模,剩下的才是"前 n 块对这步输入的加工量"——干净的"像不像上一步"信号。

## Q12: `H(t,0) − H(t−1,0) ≈ 0 → H(t−1,n) ≈ H(t,n)` 里 t 是 token id、n 是 block id?

**A: 都不是。t = denoising step 序号,n = block 计数边界;token 在张量 seq_len 维里,不在下标里。**

三个轴别混:

| 轴 | 记号 | 作用 |
|---|---|---|
| denoising step | t | 跨步比较的方向(缓存复用发生处) |
| block 深度 | n | 切 Fn/Mn/Bn 的方向(跳过发生处) |
| token | 张量内部维度 | 被一起算、一起平均,从不单独调度 |

那句话的意思:相邻两个**去噪步**的输入 hidden 差不多 → 各自过完前 n 块的输出也差不多。

## Q13: 所以缓存的对象是上一轮 diffusion loop 里的所有 hidden?

**A: 不是——一轮前向产生 28 个全尺寸 hidden(27 块 F8B0),缓存只留 2~3 个:**

```
① R(t) = H(t,8) − H(t,0)      Fn 残差,下轮判决用(指纹)
② Bn_residual = H(t,27)−H(t,8) Mn 段总变化量,下轮复用用
③ encoder 残差                 文本侧对应物
```

中间的 H(t,1)…H(t,26) 从不存储(显存里本来就逐块复用)。命中重构 `H(t,27) ≈ H(t,8) + Bn_residual` 把 19 块计算压成一个加法。显存合计 ~2 GB 恒定(H3 768p 估算);若天真缓存全部 hidden 是 28×0.9 ≈ 25 GB 每步——那才真成容量问题。设计要义:**用残差可加性把一整段网络的计算压缩成一个张量,缓存量从 O(层数) 降到 O(1)**。

## Q14: 一次缓存命中会跳过当前 step 的所有 Mn 块?

**A: 是——命中后 Mn 段所有 block 整块跳过,attention/FFN/norm 全不执行**(`call_Mn_blocks` 根本不被调用,`pattern_base.py:246-266`)。命中步成本从 27 块变 8 块 + 一次加法(F8B0 时省 ~70%)。

门卫条件(`can_cache`,`cache_manager.py:883-914`):开头 warmup_steps 强制全算;max_cached_steps 总命中上限;max_continuous_cached_steps 连续命中上限(到限强制全算刷新基准);CFG 双分支独立判决(可能一个命中一个不命中);形状变化直接判不命中。

跳过是"真的不执行"的控制流,不是低秩近似——因此对 attention kernel、序列并行、torch.compile 全透明。

## Q15: 为什么要对 Mn 区域的每个 layer 做近似判断,而不是只对 Mn 的第一个 layer 做?

**A: 前提有偏差——DBCache 每步只判一次(Fn/Mn 边界),Mn 的 19 块没有逐层判断,要么全跳要么全算。** "只判一次"正是 DBCache 的实现方式。

先纠正指纹的比较方向:R(t) 的构造在同一步内(H(t,n) − H(t,0)),但**判决是跨步的**(R(t) vs R(t−1)),不是同一步内 layer n 对 layer 0。

逐层判断的版本叫 **DBPrune**:每块独立判决、独立缓存残差,命中的块单独跳过。对比:

| | DBCache | DBPrune |
|---|---|---|
| 粒度 | Mn 段全有/全无 | 每块独立 |
| 适应性 | diff 大时整段重算 | 只重算变化剧烈的几块 |
| 判决开销 | 1 个标量 | × 块数 |
| 显存 | 2~3 个张量 | 随块数线性涨 |

逐块判决的真实收益场景:视频里前景动背景静,变化集中在深层语义块,全局判决被迫整段重算,逐块可以只重算那几块。

Bn 必算 + 下一段 Fn 必算 = 每步都有新鲜计算注入,输出轨迹仍在演化,不会"复制上一帧"。三重保险:Bn 校正 + 连续命中上限 + warmup。

## Q16: R(t) 和 R(t−1) 都减去了各自的 H(·,0),非线性在哪引入?不是抵消了吗?

**A: 非线性来自 block 内部,相减只消掉恒等直通支路。** 单块展开 `block(x) = x + f(x)`,f 含 softmax attention、LayerNorm、SiLU/GELU FFN:

```
R(t) = H(t,n) − H(t,0) = Σ fᵢ(H(t,i))   ← 全部非线性加工的累加,没被消掉
```

若 block 纯线性(block(x)=Wx),则 R(t)−R(t−1) = (Wⁿ−I)·[H(t,0)−H(t−1)],残差 diff 退化为输入 diff 的线性镜像,方案失效——质疑成立,但前提不成立。

非线性的关键后果:**同样的输入变化,在不同内容上引发的网络响应不同**。输入 diff 主要反映"调度器注入了新噪声"(无内容信息);残差 diff 反映"网络对当前内容的加工方式变了多少"(有内容信息)。各自减自己的 H(·,0) 是归一化(去共模漂移),不是抵消。

## Q17: DBPrune 里每个 layer 的 R 还是和 H(t,0) 比吗?

**A: 不是——DBPrune 判决不用残差,直接比 block 的原始输入 hidden**(`pattern_base.py:573-651`):

```
block i 输入 H_i(t) 到达
  ├─ 判决: can_prune 比 H_i(t) vs 缓存的 H_i(t−1)
  │        (buffer 名 {block_id}_Fn_original,存的是 original_hidden_states 原值)
  ├─ 命中 → H += 缓存的 {block_id}_Bn_residual,跳过 block i
  └─ 未命中 → 真算,存:输入→Fn_original(下步判决基准),残差→Bn_residual(下步复用值)
```

为什么 DBPrune 能直接比输入而 DBCache 要比残差:

- DBCache 判的是"19 块整段跳不跳"的大决定,输入 diff 被调度噪声主导,需要先过 8 块提炼"网络加工量";
- DBPrune 判的是"这一块输出会不会变"——H_i(t) 已吸收上游所有变化(含被跳块的近似),本身就是"流到这里的真实状态",顺手一比,零额外计算。
- 即:DBCache 的探针是特意保留的真算段(Fn);DBPrune 的探针就是数据流本身。

## Q18: 能否理解为:定义 logical block 区间,只看区间输入,变化小就简化?DBPrune 的区间=1 块,指纹宽度不同,单层直接看 hidden?

**A: 这个抽象成立,能统一整个设计空间。补两块拼图:**

**拼图一:指纹必须是"跑这段之前已经有的东西"**——单层只能看 hidden 不是选择而是唯一解:想用块的残差当指纹得先跑完这块(鸡生蛋)。区间宽时可以付 F 块真算换一个加工后的指纹。

**拼图二:频谱还有更极端的一端 TeaCache**:整网一个区间,指纹用 timestep embedding(零成本),判据最粗。

| | TeaCache | DBCache | DBPrune |
|---|---|---|---|
| 区间宽度 | 整个 transformer | Mn 段 | 1 块 |
| 指纹 | timestep embedding(付≈0) | Fn 残差(付 F 块) | 块输入(付 0) |

"保留多层语义"的担心,工程上不靠更精细的指纹解决,而靠两个正交手段:Bn 校正段(每步总有真算部分)和 TaylorSeer(缓存值从复制升级为外推)。完整设计空间三维:**区间宽度 × 指纹成本 × 缓存值使用方式**。

## Q19: 假如 DBCache 直接对比 H(t,8) 和 H(t−1,8) 呢?

**A: 这是合法配置(`is_l1_diff_enabled`),但默认不用,原因:**

1. **信噪比**:ΔH = ΔH(0)(调度输入漂移,共模) + ΔR(内容变化,信号)。早期 ΔH(0) 大,内容没变 diff 也虚高 → 漏判命中、白重算。残差先把共模剥掉。
2. **分母失真**:残差流累加使 |H| >> |R|,H 模式 diff 被系统性压小 → 阈值变松、不可迁移(0.08 在残差模式下跨模型还算稳定)。
3. 极端场景:latent 推进但画面静止时,R 模式正确命中;H 模式误重算。

一句话:减 H(t,0) 不是数学技巧是**信噪比技巧**——把"调度器的脚步"从"内容的脚步"里剥离。

## Q20: 这和扩散模型特点有关?早期噪声变化大几乎无法命中(输出变化也应该大);后半段 hidden 变化远比残差大导致阈值更敏感,尤其量化场景?

**A: 前半对,后半要拆成两个方向相反的效应:**

- **早期**:diff 天然大(结构形成期内容剧变)→ 必然不命中 → 恰好该真算;warmup 双保险。经验上 residual diff 曲线随步数衰减(开头 0.3~0.5,中期 0.05~0.1)。缓存收益本来就在中后期。
- **效应一(|H| >> |R|)**:H 模式分母大 → diff 压向 0 → 阈值**变松**(过度命中),不是变敏感。
- **效应二(量化)**:量化误差 ε 是绝对量级。R 模式 ε/|R| 大 → diff 有误差地板、阈值附近抖动(该命中的没命中);H 模式数值更稳但语义更钝。**敏感的是 R 模式**。
- 实践:FP8 接入 cache-dit 需重调阈值或换 H 模式;0.08 不是普适常数,依赖数值精度/模型/分辨率。

## Q21: 换个角度:前期要让 reuse 惰性(input 变化大,工程兜底强制重算),后期也要惰性(切 Bn 区),判定要灵敏所以分母要小,残差对比最好(残差不能太大否则梯度爆炸),量化友好不能数值太大?

**A: 框架成立大半,两处修正:**

1. **"两端惰性"的提炼漂亮,但两端在不同轴上**:warmup 守调度时间轴的头,Bn 守网络深度轴的尾。去噪后期(时间轴尾部)反而是命中最多、收益最大的地方;真正需要惰性的是深度轴输出端——那里直接决定像素,误差没有后续步骤弥补。
2. **"残差不能太大否则梯度爆炸"因果反了**:推理无梯度。R 小是残差网络的**训练特性**(pre-norm、残差缩放保证每块学小修正),缓存方案是白捡现成便宜,不是提出要求。
3. **"残差小所以量化友好"方向反了**:对 diff 统计,残差小是**劣势**(相对误差大、误差地板);"数值不能太大"的超量程/outlier 风险在 H 侧。量化场景下 R 模式是"语义最锐但数值最脆"。

隐藏红利:缓存残差意味着直通部分 H(t,8) 永远是当前步真值,缓存只提供小量程的"修正量",误差被天然限幅,不会在主干特征上累积。

## Q22: sglang 的 diffusion 部分,思路是一样的吗?

**A: 不只是思路一样——sglang 的 block 级缓存就是 cache-dit 本身**(钉死依赖 `cache-dit==1.3.0`,`python/pyproject.toml:114`)。

sglang diffusion(`python/sglang/multimodal_gen/`,前身 FastVideo)缓存三件套:

| | 来源 | 判决 | 粒度 | 缓存 | 显存 |
|---|---|---|---|---|---|
| cache-dit 集成 | 库依赖 | Fn 残差相对 L1(原生 DBCache) | Mn 段 | 指纹+残差 | 中 |
| TeaCache(自研 `runtime/cache/teacache.py`) | 论文复现 | timestep 调制输入相对 L1 + 模型专属多项式 rescale,累积距离超阈值才算 | 整步 | 整步 residual,CFG 正负各一份 | 最省 |
| Spectrum(自研实验 `runtime/cache/spectrum.py`) | 自研 | 不判相似度,确定性 schedule(间隔渐大) | 整步 | K 步完整 hidden fp32 环形缓冲,Chebyshev 岭回归外推 | 最大(大一个量级) |

sglang 的工程增量(`runtime/cache/cache_dit_integration.py`、`denoising.py:892`):

- 默认值重调:Fn=1, Bn=0, warmup=4, threshold=0.24, max_continuous=3——按 few-step 蒸馏模型(Z-Image)调的,比库默认激进;
- **并行一致性补丁**(line 55-118):monkey-patch 相似度,TP×SP 组内 all_reduce(AVG)——各 rank 判决必须一致否则集合通信挂死,单卡库无需考虑;
- 双 transformer(Wan2.2)适配、per-request 开关、与 CUDA graph/FSDP 互斥、与 layerwise offload 兼容;
- 互补:cache-dit 管 block 级,TeaCache/Spectrum 管 timestep 级;few-step 模型步数少,block 级 warmup 占比高,整步跳过更划算。

差异本质 = **库 vs serving 框架**的关注点:cache-dit 管判决算法(指纹/阈值/校准器),sglang 管并行判决一致性、per-request 配置、与 graph/offload/量化共存。sglang diffusion 的优化大头在别处:USP 序列并行、十几种 attention 后端(FA3/4、SageAttention、VSA、VMoBA)、NVFP4/SVDQ 量化、breakable CUDA graph、三段分离调度。支持 Wan2.1/2.2、FLUX.1/2、Qwen-Image、Z-Image、HunyuanVideo、MiniMax-H3 等约 30 个 DiT。

---

## 附:关键代码位置

cache-dit(`~/source_code/cache-dit/src/cache_dit/caching/`):
主流程 `cache_blocks/pattern_base.py:220-327`;DBPrune `pattern_base.py:452-653`;similarity `cache_contexts/cache_manager.py:552-626`;can_cache 门卫 `cache_manager.py:883-914`;校准器 `calibrators/taylorseer.py`、`foca.py`。

sglang(`~/source_code/sglang/python/sglang/multimodal_gen/`):
`runtime/cache/{cache_dit_integration.py, teacache.py, spectrum.py}`;去噪循环 `runtime/pipelines_core/stages/denoising.py`。

模型参照:`~/model-knowledge/configs/minimax-h3/`(15 个 config.json,无权重;完整快照 ~498.5 GB)。
