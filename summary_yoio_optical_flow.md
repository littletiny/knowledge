# YOIO: You Only Iterate Once — 基于多重全局信息挖掘与融合的光流估计

- 论文: [arXiv:2401.05879](https://arxiv.org/abs/2401.05879) (v1, 2024-01-11)
- 作者: Yu Jing, Tan Yujuan, Ren Ao, Liu Duo(重庆大学)
- 领域: 计算机视觉 / 双帧光流估计(Optical Flow)
- 缓存: `~/.cache/nanochat/knowledge/2401.05879/`(注意:该论文 arXiv **无 TeX 源码**,只有 PDF;文本由 pypdf 提取为 `2401.05879.txt`)

## 一句话总结

针对光流估计中**遮挡区域(occlusion)误差大**的问题,YOIO 提出用帧对(而非单帧)做"回环判断"(loopback judgment)来可靠地区分遮挡/非遮挡点并构建全局参考对,再挖掘多种全局信息(参考光流、符合均匀规律的旋转距离、occ_in 高级特征、局部 cost volume),用一个**只做一次迭代**的统一 refinement 模块融合这些信息,在 Sintel 上取得实时方法中的 SOTA:遮挡区 AEPE 比 GMA 降 >10%(occ_out 降 >15%),计算时间少 27%(53ms vs 72ms @ RTX 3090,436×1024,18.9fps)。

## 问题背景

- 遮挡点定义(同 GMA):在参考帧成像、但在下一帧不可见的 3D 点。分两类:`occ_out`(移出画面)和 `occ_in`(被其他物体/自身遮挡)。
- 遮挡点在下一帧没有对应像素,无法直接匹配,主流做法是**自相似性假设**:同一物体表面运动均匀,用与其相关的非遮挡点(noc)的光流去"猜"遮挡点的光流。
- 基于该假设的 refine 流程需要三步:① 找参考点集 P_ref(必须是 noc 子集)和待 refine 集 P_pre;② 建立全局相关对 <p_ref, p_pre>;③ 从相关对中提取足够信息做 refine。

### 现有方法的两个关键缺陷(本文动机)

1. **GMA / GMFlow 只用 frame0 单帧**自注意力选 P_ref,而判断遮挡本质需要时空信息(帧对),所以无法保证 P_ref ⊆ noc → 参考信息本身就可能是错的。
2. **refine 方式各有硬伤**:
   - GMFlow:直接把 p_ref 的光流值赋给 p_pre,计算省但精度差——尤其**旋转运动**下同物体两点距离越远光流差越大,直接拷贝误差随距离增大;
   - GMA:基于 RAFT 多次迭代,信息从 noc 传播到远处 occ 需要足够多迭代,耗时随迭代数线性增长;且局部视野全被遮挡时缺少全局方向指引。

## 方法:YOIO 三组件

整体建立在 GMFlow 之上:先用 GMFlow 管线(CNN 提 1/8 特征 → Transformer 增强 → 全局匹配 softmax)得到初始光流 Flow0(Flow0 在 noc 区准、occ 区不准),然后:

### 1. 多重全局信息提取模块(Sec 3.3)

核心是 **loopback judgment algorithm**,基于两条原理:
- 原理1:全局上下文中,一个点在任何帧中与"它自己"最相似;
- 原理2:除自身外,与它第二相似的是它的自相似点。

算法:对 frame0 中任意点 p0,用全局特征在 frame1 找最相似点 p1(实现上就是 `Flow0 + init_coord`,见论文式(1)(2));再用 p1 对 F1 线性采样构造 F1′,F1′ 与 F0 再做一次全局匹配得到 Flow_ref,反查回 frame0 得到 p0′:
- 若 p0′ == p0(回环成功)→ p0 必为 noc 点(反证法可证);
- 否则 p0 是 occ 点,且 p0′ 大概率是 noc 点,可直接作为 p0 的参考点,构成全局相关对 <p0′, p0>。

由此顺带得到三个产物:
- **Occlusion map**:`‖Flow_ref‖` 在该点是否为 0 即判定是否遮挡。**不需要遮挡标签训练**(Flow_ref 与 Flow0 用同一套特征和匹配,只需光流标签监督),也**不需要算双向光流**;GMFlow 转置相关矩阵的方案要 `3G + softmax(4D 矩阵) + 额外开销`,本方法只需 `2G`(G = 一次全局匹配)。
- **Ref-flow**:用 Flow_ref 从 Flow0 采样得到的参考光流。
- **符合均匀规律的距离**:旋转运动下直接用 p_pre 与 p_ref 的欧氏距离不合理(图5:AB 过旋转中心而 AC 不过,欧氏距离差与光流差不满足均匀关系)。改为用**点到旋转中心 O 的距离差** `| |AB| − |AO| |`(式5);旋转中心坐标由参考点及其邻域在两帧中的坐标(用 Flow_ref 采样)经卷积网络回归得到。

### 2. occ_in 高级特征图(Sec 3.4)

occ_in 点特殊:训练时为最小化 loss,它会匹配到"遮挡它的那个 noc 点",二者全局特征相似、Flow0 中已经"准",但二者**不属于同一物体**,不满足自相似假设,形成的相关对是错的,必须单独识别处理。
判别依据:occ_in 对 **全局特征相似但局部特征不同**(正常相关对两者都相似)。做法:局部特征 f0·f1 点积得局部相似图,全局特征 F0·F1 点积得全局相似图,拼接后过卷积网络提取高级特征供后续模块识别 occ_in。

### 3. 统一 refinement 模块(Sec 3.5–3.6)

- 另外仿 RAFT 构建**局部 cost volume**(以 Flow0 为初值,相关半径取 3——因为 GMFlow noc 区 AEPE < 1),进一步提升 noc 区精度;
- 把上述所有信息(参考光流、旋转距离、occ_in 高级特征、局部 cost volume)拼接后送入一个卷积网络,**只迭代一次**;
- 网络输出 weight 和 residual,最终光流 = `(1−weight)·Flow0 + weight·residual`(式6)——weight=0 时退化为不修正,正好对应 occ_in 点。
- 监督方式与 GMFlow 相同(光流标签)。

## 实验结果

- 训练流程:FlyingChairs(100K iter, bs16, lr 4e-4)→ FlyingThings3D(800K iter, bs8, lr 2e-4)→ Sintel 混合集(KITTI+HD1K+Things+Sintel,200K iter)微调;优化器 AdamW;骨干与 GMFlow 相同(6 个 Transformer block),RAFT convex upsampling。
- **vs GMFlow(1/8)**:Sintel clean noc 区 Flow0 AEPE 0.69 vs 0.89(相对提升 22.5%)。可视化显示旋转运动下 GMFlow 离旋转中心越远误差越大,YOIO 几乎无此现象。
- **vs GMA**(Sintel train,表3):YOIO 一次 refine 即在 occ 区全面优于 GMA 12+ 次迭代——clean: occ 10.58→9.51(+10.1%),occ_out 12.52→10.62(+15.2%);final: occ 17.33→15.86;albedo: occ_out 10.22→8.94(+12.5%)。代价是 noc 区略差(0.58→0.67,因为只做一次局部 refine),整体 All 基本持平(1.30 vs 1.32),但**速度快 27%**(53ms vs 72ms)。
- **Sintel 在线 benchmark**(表4,C+T+S+K+H):all AEPE 1.365,优于 GMA(1.391)、GMFlow 1/4(1.74)、RAFT(1.94)等;unmatched 7.584 为最优——**实时方法中的 SOTA**。
- **消融**(表5):不加距离 occ AEPE 9.69 → 加欧氏距离 9.55 → 加"符合均匀规律的距离" 9.51,验证距离设计必要且旋转中心距离更优。

## 局限

- 全局特征提取依赖 GMFlow 的 Transformer,训练/测试数据差异大时(合成 Things → 真实 KITTI)泛化仍不够好;
- noc 区精度不如多迭代方法(单次 refine 的天然代价),作者留作后续工作。

## 与本工作区(~/source_code/)的关联

这是一篇**光流/视频理解**论文,与 LLM 推理基础设施没有直接关系,但有几处可借鉴的思想:

1. **"回环一致性"判别思想**:前向匹配再反向匹配、回环成功才采信——这与 LLM 推理中一些自校验思路同构(如 speculative decoding 的 draft-verify 回环、KV cache 选择中用双向信号验证相关性,参见 `summary_kv_selection_signals` 类讨论)。本质上都是"单程信号不可靠,用闭环一致性替代显式标签"。
2. **一次迭代 vs 多次迭代的信息论视角**:YOIO 的论点是"把全局信息一次性挖够、融合好,就能省掉迭代"。这对推理系统设计有启发:很多迭代式 refine(多轮 attention 修正、多轮 draft)的瓶颈是每轮视野/信息不全,若能在一步内把必要的全局上下文聚合好,延迟可线性下降。
3. **视频生成方向**:若未来涉及视频生成模型(如 `~/model-knowledge/configs/minimax-h3/` 的音视频生成流水线),光流/遮挡建模是运动一致性、帧间 warp 类方法的基础组件,GMA/GMFlow/RAFT 这一系列文献值得知道。
4. 工程细节:occlusion map 无需额外标签、无需双向光流,只用 2 次全局匹配替代 `3G + 4D softmax`——典型的"复用已有计算图产物、避免重复大张量运算"的推理友好设计。
