# Gray Failure: The Achilles' Heel of Cloud-Scale Systems (HotOS 2017)

- 作者: Peng Huang (Microsoft Research / Johns Hopkins University), Chuanxiong Guo, Lidong Zhou, Jacob R. Lorch (Microsoft Research), Yingnong Dang, Murali Chintalapati, Randolph Yao (Microsoft Azure)
- 出处: HotOS '17, Whistler, 2017 年 5 月, 6 页, DOI 10.1145/3102980.3103005
- PDF: https://www.microsoft.com/en-us/research/wp-content/uploads/2017/06/paper-1.pdf (USENIX 原始链接已 404)
- 性质: 经验 + vision paper —— 基于 Azure 生产事故一手经验,首次对"灰色故障"给出形式化定义(**differential observability,差异性可观测**),指出传统 fail-stop 容错在云规模下的系统性盲区,并勾画解决方向。

## 核心概念:灰色故障 (Gray Failure)

**动机**:云的冗余资源只有在故障能被**检测**到时才有用。云环境中主要的可用性崩溃和性能异常,往往不是 fail-stop 故障,而是表现微妙、难以快速确诊的灰色故障:严重性能降级、随机丢包、时好时坏的 I/O、内存抖动、容量压力、非致命异常等。作者的一线经验是:**Azure 大多数事故背后都是灰色故障**;且规模越大、多租户交互越复杂,灰色故障越常见。

**关键洞见 —— differential observability(差异性可观测)**:灰色故障的本质特征是**不同实体对系统健康状况的观测不一致**:app(服务的使用方)已经受害,但系统内部的 observer(故障检测器)认为系统健康,于是 reactor(恢复组件)不会动作。例子:请求处理模块卡死但心跳模块正常 → 基于心跳的错误处理模块认为系统健康,客户端却看到故障;链路带宽显著下降但连通性正常 → 连通性测试无异常,应用性能却已劣化。

**抽象模型(§3,Figure 2)**:两个逻辑实体 —— system(提供服务)和 app(使用服务);system 内含 observer(主动/被动收集健康信息)和 reactor(据此恢复)。一个 system 可以是另一个 system 的 app(如网络为存储服务提供传输)。定义:**当至少一个 app 观测到系统不健康、而 observer 观测到系统健康时,系统正处于灰色故障**。

**四象限(Table 1)**:

| | app 观测好 | app 观测坏 |
|---|---|---|
| **observer 观测好** | ➊ 无故障(或良性潜伏故障) | ➋ **灰色故障**(用户受害、无人修复) |
| **observer 观测坏** | ➌ 良性差异(主动预警;误报是另一类问题) | ➍ 双方一致,故障将被修复(crash/fail-stop 属此类) |

**时间演化(§3.3)**:典型路径是 **潜伏故障(latent)→ 灰色故障(degraded,app 可见而 observer 不可见)→ 完全故障(observer 也终于看到)**,即在四象限中从 ➊ → ➋ → ➍;内存泄漏是典型例子。间歇性灰色故障中这个转移反复发生。➋ 阶段持续的时间就是"用户受害而系统无感知"的窗口(Figure 3 的 VM 案例里,compute manager 的红点全部来自**用户手动重启**,真正的根因很久之后才被发现)。

## 四个 Azure 生产案例(§2)—— 容错机制与灰色故障的反直觉互动

1. **高冗余反而降低可用性(Clos 网络)**:交换机 crash 会被路由协议绕开,但**随机静默丢包**这类灰色故障不会触发 reroute。前端 fan-out 到 m 个后端、需等几乎全部响应的负载模式下,某条请求路径经过特定 core switch 的概率是 1−((n−1)/n)^m,m 大时趋近 100% —— 任一 core switch 灰色故障几乎拖慢**每一个**前端请求。core switch 越多(冗余越高),至少一台处于灰色故障的概率越大,可用性反而越差。灰色故障迫使我们重估"冗余提升可用性"的常识。
2. **故障检测器的盲区(VM 网络故障)**:VM 内因驱动 bug 出现严重网络故障,但远端 compute manager 通过 host agent 的本地 RPC 收心跳,**不走 VM 的外部网络路径**,因此毫无察觉;直到用户报障才开始恢复,受害与感知之间存在长时间差。缺的是让 manager 观测 VM 内部状态的通道。
3. **"恢复"变成"杀死"(Azure Storage)**:某 data server 严重容量受限,但资源上报 bug 使 storage manager 不知情,继续路由写请求 → server 崩溃重启 → 重启不治本 → 再次压垮 → 故障检测器最终判定其"不可修复"并摘除 → 副本工作流的另一处微妙问题导致总可用存储下降 → 压力转移到健康服务器 → 更多服务器重蹈覆辙 → **级联故障**。reactor 的恢复动作(重启)本身成了维持故障的燃料。
4. **甩锅游戏(blame game)**:VM 因存储/网络问题访问不到虚拟磁盘而 crash,若无故障检测器看到存储/网络侧的问题,compute 集群的检测器会**错误归因**于 VM 计算栈;各子系统团队互相甩锅,无人握有根因证据。

## 解决方向(§4)

1. **弥合观测鸿沟(§4.1)**:从单一的 fail-stop 检测(心跳)转向**多维度健康监控** —— 类比人体检查不只看心跳,还要看体温、血压。如对 VM 案例,利用 VM 内性能计数器(网络计数器在触发事件后骤降,见 Figure 3)提前发现连通性问题。
2. **逼近 app 视角(§4.2)**:让系统完全测量每个 app 的观测不可行(多租户、模块化边界),可行的是测量**近似 app 观测**的指标,如用 Pingmesh 式主动探测模拟常见应用看到的 server-to-server 延迟/可达性。注意陷阱:**过度主动探测会加重已降级系统的负担**;且 app 是逻辑实体(可能是另一个 system),逼近 app 视角不等于简单的端到端测量。
3. **利用规模的力量(§4.3)**:灰色故障常源于单个 observer 的局部视角,**聚合大量互补组件的观测**可快速暴露它 —— 许多案例**只有分布式观测才能发现**(intrinsic gray failure)。设计问题:聚合/推断放在哪?太靠近系统核心会限制可见范围,太靠近 app 则系统内建的容错掩蔽会让差异暴露太晚;作者设想一个**独立于核心系统边界之外、但与 observer/reactor 相连的独立平面**。全局探测 + 统计推断还能定位"偶发但持续"的故障设备(单个随机丢包事件难以归因),并通过聚合 VM 磁盘故障事件与拓扑信息解决甩锅问题(实践中已用于定位存储过载和 ToR 交换机意外重启)。
4. **利用时间模式(§4.4)**:灰色故障的前奏通常是 observer 认为"太小不足以报障"的潜伏故障;识别其时间演化模式可提前预警(但要小心误报,多数潜伏故障是良性的)。灰色故障发生后仍有**机会窗口**:Figure 4 中 compute 侧在 t1 就看到 remote I/O exception 上升,而 storage 侧到 t2 才察觉并行动 —— t1 到 t2 之间跨子系统关联差异性观测,本可阻止级联故障。

## 与 HotOS'21 Metastable Failures 框架的关系

- 同属"**无硬件/软件 bug 直接崩溃、却造成 outage**"的故障家族,视角互补:gray failure 刻画**检测维度**(observer 与 app 的观测分歧,故障"看不见"),metastable failure 刻画**动力学维度**(trigger + sustaining effect 的反馈环,故障"停不下来")。
- 两者可以叠加:案例 2.3 中 storage manager 看不见容量灰色故障 → 反复写入 → 崩溃 → 重启 → 再写入,正是"灰色故障(看不见根因)喂养亚稳态反馈环(重启维持效应)"的复合形态;failover/摘除节点导致容量缩减、压力传染,也是 metastable 论文中"failover 是重试近亲"的具体实例。
- 防御上互补:gray failure 要求**多维度/分布式观测、逼近 app 视角的健康监控**(回答"怎么看见"),metastable 要求**识别并削弱维持反馈环、以 hidden capacity 规划容量**(回答"看见后怎么不停不下来")。metastable 论文的 characteristic metric(排队延迟等)正是 gray failure 意义上"逼近 app 观测"的指标选择。

## 启示

- **故障模型假设决定容错有效性**:心跳/健康检查只覆盖 fail-stop;"进程活着但服务已坏"是常态。LLM 推理系统同样:一个 prefill/decode 实例可能健康检查通过但已陷入显存碎片化、KV cache 损坏或 NCCL 半卡死状态,调度器(observer)毫无察觉地把请求继续打过去。
- **冗余不是免费的**:vLLM/SGLang 的多副本、Mooncake 的多路径 KV 传输,在"fan-out 等待全部响应"模式下(如 TP/PP 同步、MoE all-to-all、分布式 KV 拉取),任何一个灰色节点都拖慢整体 —— tail latency 由最慢副本决定,冗余越多踩中灰色节点的概率越大(对应 §2.1 的概率公式)。
- **观测要逼近 app 视角**:推理系统的"app 观测"是 TTFT/TPOT/end-to-end 延迟和 token 吞吐;仅监控 GPU 利用率、进程存活远远不够。应像 Pingmesh 一样做合成探针请求(canary request),并以排队延迟类指标为特征指标,与 metastable 的防御建议合流。
- **警惕"恢复"成为燃料**:PD 分离架构下,若调度器看不见某 prefill 节点的灰色故障(如 KV cache 传输慢)而持续派发,超时重试 + failover 摘除会把负载压到剩余节点,复刻案例 2.3 的级联。恢复动作(重启、迁移、摘除)前应先确认观测到的是根因而非症状。
- **跨层观测关联防甩锅**:推理栈跨越调度器、推理引擎、KV 传输层、网络、存储,灰色故障的症状(超时、延迟毛刺)与根因常不在同一层;聚合各层观测(如把 KV 传输异常事件与拓扑/节点事件关联)才能避免"甩锅游戏",并抓住 t1–t2 的机会窗口在级联前介入。
