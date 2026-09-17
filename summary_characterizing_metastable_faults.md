# Characterizing Metastable Faults and Failures (arXiv 2026)

- 作者: Ali Farahbakhsh, Lorenzo Alvisi, Robbert Van Renesse (Cornell); Qingjie Lu, Andreas Haeberlen (UPenn)
- 出处: arXiv:2606.00942v2, https://arxiv.org/pdf/2606.00942 (HTML: https://arxiv.org/html/2606.00942)
- 性质: 理论/形式化 paper —— 对"亚稳态故障"给出第一个**因果性 (causal)** 刻画:把根因定位为结构性的**亚稳态缺陷 (metastable fault)**,批评以往工作只从"自持过载"这一症状出发。配套方法论(预测故障 + 设计亚稳态缺陷容错 MFT 系统)和 DSL 工具 Nyx。

## 与 HotOS'21 框架的关系

HotOS'21 (Bronson 等,见 `summary_metastable_failures.md`) 及后续工作 (OSDI'22 "Metastable Failures in the Wild" 等) 是**现象学 (phenomenological)** 的:从症状(正常负载下高延迟、低 goodput)出发,把亚稳态等同于"自持过载 (self-sustaining overload / congestive collapse)",把根因归为负载放大或容量退化。本文指出这条路线有三大缺陷:

1. **无因果洞察**:治的是"发烧"不是"病毒"。以 retry storm 为例,根因不是单纯的"重试放大负载",而是**负载放大与容量退化之间的环形相互作用**,由 (i) 组件的功能组合结构和 (ii) 各自的调度策略共同塑造。
2. **补救措施不到位**:不知根因时标准动作是"卸负载 (shed load)",但本文分析表明真正的问题在**调度**:retry 侧换指数退避可消除故障;更好的是改 server 的被动调度(请求与重试分队列、请求优先),没有退避的延迟代价。
3. **覆盖太窄**:无法解释不表现为高延迟/低 goodput 的亚稳态,例如本文新案例(§7)中表现为**活跃服务器数量持续振荡**的事故。

本文把 fault(结构性缺陷)与 failure(动态表现)分开:**fault 是必要的但不是充分的**;fault 加上"偏袒去稳定化动作的调度"才酿成 failure(Theorem 1:有 failure 必有 fault)。

## 核心概念与形式化(§3–§5)

**系统模型**:系统 = 状态机 + 环境;组合 (composition) 产生 writes-to 关系,导出 **composition blueprint**(组件间写依赖有向图)。假设组合中的联合动作是**可串行化 (serializable)** 的,否则组件的"势"无法良好定义。

**势函数 (potential function, Def 1)**:f: Σ→R≥0,f(s)=0 当且仅当 s 是好状态;系统自身动作只能维持或降低 f——**去稳定化的唯一来源是环境**。

**稳定化系统 (stabilizing system, Def 2)**:源自 Dijkstra 自稳定概念。存在环境谓词 E,只要环境足够"乖"(维持 E),系统终将达到并保持 f=0,记作 E ⇝+ f=0(assume-guarantee 形式)。

**兼容性 (compatibility, Def 3)**:组合健全性检查——假设某组件的所有写邻居都已稳定,它自己也能在良性环境下稳定。不兼容的组合是"本就不该组合"的,排除出亚稳态讨论。

**去稳定化动作 (Def 4)**:组件 S₁ 写给 S₂ 的动作 α,若在某状态下(存在兼容环境)使 f₂ 不减(为正时)或从 0 变正,则 α 是去稳定化的。

**亚稳态缺陷 (metastable fault, Def 5)**:兼容的稳定化系统组合中,blueprint 里存在一个**环**,环上每个组件写给下一个组件的某个动作是去稳定化的。即:**每个组件单独都在努力自我稳定,但各自的稳定化动作恰好去稳定化对方,形成"组合之罪" (sins of composition)**。定位 fault 是局部的:只需逐对组件检查动作是否去稳定化,可用 model checker 自动化。

**亚稳态故障 (metastable failure, Def 6)**:存在兼容环境 E 和执行 σ,使得环境始终兼容 (□E) 而系统**无限次**出现正势 (□◇¬(∧fᵢ=0))。fault → failure 的关键是**调度**:调度器(队列、OS、定时器等一切决定事件顺序的部件)偏袒去稳定化动作 → 势保持为正 → 更多去稳定化事件 → 更差的调度决策,自我强化。反之,若稳定化动作被优先足够久,兼容性会驱动全局势归零,且归零后被推迟的去稳定化动作会自然消失。

**Theorem 1(fault 是 failure 的必要条件)**:对组件数 n 归纳证明(无 fault ⇒ 任意兼容环境下全局势最终归零)。这推广了"反馈环是亚稳态必要条件"的已有结论,并给出一个**可自动验证的排除标准**:构造 blueprint、检查是否存在去稳定化动作环;无环则(在势模型假设下)不可能发生亚稳态故障。

## Retry storm 贯穿示例(§2)

Nyx 模型:server 有 35 单位 CPU/步(服务容量 s=35),retrier 超时 T=4 步,名义负载 r=30。

- server 的势 f₁ = max{0, Q−s}(过载量);retrier 的势 f₂ = max{0, P−P_threshold}(pending 积压,分析给出 P_threshold = (s−r)·T)。
- fault:retrier 的 remove-n-retry 可能不降 f₁(重试涌入);server 的 serve 可能不降 f₂(服务的是已答过的重试,ack 无效)——"server 浪费容量服务重试" 与 "retrier 放大负载" 互为因果。
- failure:retrier 每 T 步激进重试 + server 单队列 FIFO 的被动调度,让重试逐渐主导队列,容量被无效服务侵蚀,循环加深。
- 三种场景演示:无冲击 / 安全冲击(短暂扰动,甚至因负载升高出现 goodput 小峰)/ 不安全冲击(pending 无界增长,goodput 崩溃)。

## MFT 方法论(§6)

四步,目标是在**不消除 fault** 的前提下证明系统亚稳态缺陷容错 (metastable-fault-tolerant):

1. **提取亚稳态骨架 (metastability skeleton)**:只保留与亚稳态相关的操作模型——队列(调度请求)、资源(调度任务/线程)、定时器、请求类型间的因果链。Nyx DSL 直接暴露这些抽象,可把 OS 调度等跨栈细节纳入同一模型(生产代码里它们散落在不同代码库)。
2. **研究动力学**:用**向量场 (vector field)** 可视化系统在"势空间"中的倾向。亚稳态表现为两种共存倾向:一个拉向零点、一个发散到正势。与 Metafor 用连续时间马尔可夫链(无记忆)不同,Nyx 认为**有亚稳态的系统有记忆**(记忆影响调度),因此用实际执行的 lock-step 轨迹作为 ground truth 生成向量场(记录每个状态的 next(s) 集合,取向量为平均倾向);设计者可用 `inject` 等构造指定冲击。
3. **管理调度**:两条原则——**(R1) 试探性地调度去稳定化动作;(R2) 在组合稳定之前,稳定化动作优先于去稳定化动作**。R2 保证收敛,R1 + 兼容性保证稳定后不再丢稳定(被推迟的去稳定化动作会消失)。指数退避是 retry storm 场景下同时实现 R1/R2 的一种方式。
4. **完成证明**:指定 adversary(冲击必须有限:存在未知时刻之后对抗停止且后果被修复,类比共识的 partial synchrony 假设),证明 (i) 势不增、(ii) 势有时减、(iii) 全局稳定后无去稳定化事件。**对某个 adversary 证明稳定化 = 对该 adversary 能触发的所有 fault 容错**——一次证明覆盖一大类 fault。论文展望:亚稳态研究应像安全研究一样,由新 adversary 暴露新漏洞。

## 案例研究(§7 + 附录 B)

### 案例二(新案例):Oscillating Membership——大型游戏平台集群管理器事故

某数亿用户商业游戏平台的内部 post-mortem:集群管理器 (CM) 维护至少 W_min 个活跃 worker,靠心跳超时判定,超时则发 sleep(给失联 worker) + wakeup(给空闲 worker)。事故表现为**活跃 worker 数 W 持续振荡**(crash 恢复后仍振荡,且与丢包/排队延迟无关),不是过载症状——HotOS'21 框架无法解释。

- **两个隐性元凶**:(i) 过载的活跃 worker 会**漏心跳**(服务线程饿死心跳线程);(ii) **wakeup 比 sleep 慢**(要 provision VM,sleep 只是状态变更)。
- **循环**:部分 worker crash → 剩余 worker 过载 → CM 发 wakeup → 新 worker 还没醒来,过载 worker 漏心跳被 sleep → 新 worker 醒来时老 worker 已闲,新 worker 又过载 → 无限重复。
- **势函数**:worker f₁ = W_noOverload − W(还差多少活跃 worker 才不漏心跳);CM f₂ = W_min − W′(CM 认为的活跃数缺口;CM 无法靠自己的动作降 f₂,依赖及时心跳)。兼容性成立;fault 是 worker 漏心跳去稳定化 CM、CM 的 sleep 去稳定化 worker 构成的环。
- **failure 的调度根源**:sleep/wakeup 同时发出但 sleep 先生效,这个时间差里剩余过载 worker 继续漏心跳,引发更多 sleep,雪崩式漂移。
- **MFT 设计 CMmft**:不需要事先知道"worker 会饿死心跳"这个 fault。worker 状态细化为五态 {idle, active, waking-up, snoozing, pending-timeout};为 sleep 引入**人工延迟 d′_sleep,要求 d_wakeup + T < d′_sleep**(只保证 d_wakeup < d′_sleep 不够——那只是每对 wakeup/sleep 有序,无关的 sleep 仍可能插队)。证明三个不变量(反向证明):(iii) W=W_min 后不再发 sleep;(ii) U = |active|+|waking-up|+|pending-timeout| ≡ W_min,故总有足够 worker 在唤醒途中;(i) 任意两个 waking-up 的唤醒时刻差 ≤ T,所以 sleep 必排在所有 pending wakeup 之后。向量场:40 worker 集群、1000 次执行 × 100 秒,约 3 分钟生成,双倾向一目了然。

### 案例三:Look-aside Cache(附录 B)——展示"直接消除 fault"

- 势:cache f₁ = C_max − C(缺 key 数);webserver f₂ = pending 数;database f₃ = max{0, Q−s}。存在两个 fault 环:cache↔webserver(超时丢请求 → 不回填 cache → cache 永远冷)和 webserver↔database(FIFO 队列 → 首个超时后**所有更年轻的请求也必然超时**,backlog 稳定自持)。
- **消除 fault(B.1)**:webserver 在每 s 个发向 DB 的请求中随机标记一些**永不超时** → 总能回填 cache → webserver 的动作总能降 f₁ → 环被打破,永不再发生该故障。
- **或用调度容错(B.2)**:加大超时 T 让 cache 有时间回暖;或 DB 改为**优先服务年轻请求**(降低其超时概率)。正确性证明归结为一场"cache 回填速度 vs 首个超时"的赛跑。

### 相关工作定位(§8)

- Bronson et al. (HotOS'21) 首次提出;Huang et al. (OSDI'22) 归纳工业事故为"负载放大/容量退化";Habibi et al. 用排队论+CTMC 分析 retry storm;Isaacs et al. (Metafor) 提供模拟器与工具但都未刻画根因,也无法覆盖非过载型亚稳态。
- 被以往忽略的亚稳态实例:Floyd & Van Jacobson 的路由器同步;Khan et al. 的负载均衡器↔热管理器恶性循环(AC 故障 → CPU 过热 → 热管理器让 worker 睡眠 → LB 误判空闲加负载 → 持续过热);Hadoop 心跳与恢复动作互相干扰;Ford 的 LB 与功耗优化器振荡。
- 与 deadlock 的类比:都源于组件间循环依赖而非局部错误,都可用图刻画,解法都在调度。
- 与 Anvil/ESR (OSDI'24) 的区别:ESR 假设组件在控制器更新状态期间停止去稳定化它——而亚稳态恰恰发生在这种干扰持续存在时。

## 启示

- **fault ≠ failure,调度是导火索**:结构性缺陷(互相去稳定化的环)可以长期潜伏,只有"偏袒去稳定化动作的调度"才把它点着。这解释了 HotOS'21 观察到的"系统能在脆弱态运行数年不出事"——也意味着容量/负载视角之外,**审查事件排序(队列策略、超时、定时器、线程调度)是独立的防御维度**。
- **两条通用防御路线**:(R1/R2) 去稳定化动作要"试探性、可推迟",稳定化动作要优先,直到全局稳定——指数退避、请求/重试分队列优先原请求、sleep 加人工延迟,都是同一原则的实例。在 LLM 推理系统里对应:过载时 prefill/decode 请求与"重试/重发"流量分队列;KV cache 重建(稳定化)优先于新请求接入(可能去稳定化);PD 分离场景下节点重启/重均衡命令的执行顺序需要像 CMmft 那样显式保证"恢复类动作先于下线类动作生效"。
- **症状思维会漏掉非过载型亚稳态**:活跃实例数振荡、auto-scaling 抖动(扩容决策比生效快/慢不对称,正是"wakeup 慢于 sleep"的同构问题)在推理集群中同样常见。监控不应只盯延迟/goodput,振荡本身(如 `~/source_code/vllm/`、`sglang/` 的自动扩缩容回路、mooncake 的 KV 节点上下线)就是亚稳态特征指标。
- **"一次证明覆盖一类 fault"的 adversary 思路**很实用:不必事先找到具体缺陷(游戏里 fault 藏在"心跳线程被饿死"这种实现细节里),只要对 adversary(有限冲击)证明调度策略 R1/R2 成立即可。类比到推理系统:可以不要求知道 KV cache 丢失的确切成因,只要求调度器保证"cache 回暖动作不会被新流量饿死"。
- **势函数 + 向量场是可操作的设计工具**:给每个组件找"距离好状态多远"的标量(队列过载量、pending 数、活跃实例缺口、cache 缺失率),两两检查动作是否去稳定化,再画向量场看是否双倾向共存——比全系统压测便宜,且能在设计期暴露问题。Nyx 的教训是系统**有记忆**,无记忆的马尔可夫模型会漏掉调度相关动力学;对 KV cache 这类强状态系统尤其如此。
- **能消 fault 就消 fault**:look-aside cache 的"随机请求永不超时"一行改动直接破环——对应 HotOS'21 的"read-through cache 没有死结"。形式化把这类修复从"事后经验"变成"从定义推导出的设计决策"。
