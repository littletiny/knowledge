# Metastable Failures in the Wild (OSDI 2022)

- 作者: Lexiang Huang (Penn State / Twitter), Matthew Magnusson*, Abishek Bangalore Muralikrishna* (UNH, *共同一作), Salman Estyak, Rebecca Isaacs (Twitter), Abutalib Aghayev, Timothy Zhu (Penn State), Aleksey Charapko (UNH)
- 出处: OSDI '22, https://www.usenix.org/conference/osdi22/presentation/huang-lexiang ;PDF: https://www.usenix.org/system/files/osdi22-huang-lexiang.pdf ;复现实验代码: https://github.com/lexiangh/Metastability
- 性质: 实证研究 —— HotOS '21《Metastable Failures in Distributed Systems》(Bronson 等,总结见 `summary_metastable_failures.md`)的直接后续。Bronson 等只给了 Facebook 的简化示例并断言该模式普遍,本文从数百份公开事故报告中实证确认了亚稳态故障的普遍性(21 个事故 + 1 个 Twitter 内部案例),并把框架形式化为带负载/容量符号的模型 + 4 个定理,再用 3 个受控实验复现不同类型的亚稳态故障。

## 核心发现(三大结论)

1. **亚稳态故障普遍存在**:详查 21 起跨 11 家组织的公开事故(摘要按 22 起计,含 Twitter 案例),含 AWS 4 起、Google Cloud 4 起、Azure 4 起,以及 IBM、Spotify、Elasticsearch、Wikipedia、CircleCI、Cassandra、Facebook 等。
2. **亚稳态故障是重大事故的常客**:AWS 近十年 15 起重大事故中至少 4 起是亚稳态故障(含 2021-12-07 us-east-1 大瘫痪 AWS4,影响航司、智能家居、支付系统)。事故时长 1.5~73.53 小时,4~10 小时最常见(35%)。
3. **扩展模型**:区分两类 trigger 和两类维持(放大)机制,并用实验证实"脆弱态不是二元的"——是否跌入亚稳态由当前脆弱程度、trigger 强度、trigger 持续时间共同决定。

## 方法论与事故统计(§2)

- 数据源:云厂商官方事故页(AWS/Azure/GCP/IBM 状态历史)+ 小型公司 postmortem 社区、周报。找"亚稳态征兆":临时 trigger、work amplification/维持效应、以及需要大幅 load shedding 才能恢复的缓解过程。
- **Trigger 统计**:约 45% 源于工程师错误(buggy 配置/代码部署、潜伏 bug);约 35% 是负载尖峰;**45% 的事故有多个 trigger**;约 50% 的 trigger 有直接人为因素(部署、测试不全、日常维护)。
- **维持效应统计**:retry 是最常见的维持效应(>50% 事故);其他有级联过载、昂贵错误处理(如 SPF2 的过量错误日志)、锁竞争、leader election churn(ELC1 的 ZK 抖动)、GC。
- **缓解手段统计**:直接 load shedding(限流/丢请求/改负载参数)用于 >55% 的事故;间接手段包括重启清队列、策略变更(如 CAS1 关掉某 feature 让节点能重新入群)、扩容。

## 形式化模型(§3)

以负载 L(单位 RU/s,每请求消耗若干 RU)和容量 C 为基础:

- **Lorg / Corg**:有机负载/容量——含 trigger 的瞬时效应,不含亚稳态放大;**Lsys / Csys**:含放大后的实际负载/容量。不过载要求 Lsys < Csys。
- **两类 Trigger**(Definition 1):**load-spike**(Lorg 突增,幅度上界 mtrigL)与 **capacity-decreasing**(Corg 下降,幅度上界 mtrigC;如机架故障、部署了低效代码)。这是相对 HotOS '21 的第一点扩展——容量下降而负载不变同样能触发。
- **过载条件**(Theorem 1):mtrigL + mtrigC ≥ Cnorm − Lnorm 才可能过载;无过载 trigger 则永无亚稳态故障。
- **两类维持放大**(Definition 4):
  - **Workload amplification**:αL(t) = Lsys/Lorg ≥ 1。两个机制:请求数变多(retry)或单请求变贵(昂贵错误处理/日志)。retry 类放大通常有**放大延迟**(等超时到期才开始),短超时对瞬时故障友好但会更快点燃放大环(AWS2 的短握手超时是帮凶)。
  - **Capacity degradation amplification**:αC(t) = Csys/Corg ≤ 1。如队列堆积 → GC 变忙 → 处理变慢 → 队列更堆积。特例是 **sustained degradation**:容量持续降级(trigger 消失也不回),如 HotOS '21 的 look-aside cache 死结(cache 因超时永远填不回去)。
  - 放大上界 wL(Δttrig)、wC(Δttrig) 是 trigger 持续时间的单调增函数(1 → w\*),即 **trigger 拖得越久,放大越大**。
- **四个场景**(Figure 1):2 类 trigger × 2 类放大,实践可叠加出现。
- **稳定态**(Theorem 2):**Cstable = Cnorm / (w\*L · w\*C)**;Lnorm < Cstable 则触发最大放大也能自愈(CAS2:Cassandra 只跑在 10–30% 容量,巨大 trigger 下仍自行恢复)。
- **脆弱度**(Theorem 3):wL(Δttrig)·wC(Δttrig) < Cnorm/Lnorm 则不会跌入亚稳态。脆弱度由 Cnorm/Lnorm(headroom)、放大系数、trigger 时长共同决定;放大延迟是给运维争取时间的第一道缓冲。
- **亚稳态边界**(Theorem 4):当过载量 Lsys(t) − Csys(t) ≥ αL·mtrigL + αC·mtrigC 时,即使撤掉 trigger 系统仍过载 = 已进入亚稳态。**实践要点:在过载量越过此边界之前采取更激进的措施**。
- **恢复**(§3.6):修 trigger 只是第一步,进入亚稳态后必须打破维持效应。两条路:(a) load shedding 把负载压到 Cstable 以下(>50% 事故采用,但不懂反馈环就不知道要削多少,导致拖延和重启服务器等破坏性动作);(b) 抬 Cstable——降放大上界(如给 retry 次数封顶:最多重试 2 次 → 放大 ≤3x;不封顶 = 没有稳定区)或扩容抬 Cnorm。

## Twitter 内部案例:GC 亚稳态(§4)

成熟核心服务(多年良好运维)在忙时做峰值压测触发:负载尖峰(48 分钟处)→ 队列变长 → 堆内存压力与 mark-and-sweep 工作量上升 → GC 变忙抢占 CPU → 处理变慢 → 队列更长,反馈环闭合。运维在 83/106 分钟两轮 load shedding,负载降到压测前水平后 **SR 仍持续跌破 SLO,直到重启服务才恢复**。佐证:第二次降载期(106–118 分)负载比 40 分钟处低 20%+,GC 却更忙、队列高 50%+。正常 3 天数据中队长 vs GC 时长、GC 时长 vs 延迟均正相关(并控制了负载变量)。事后工程师的三种修复(优化后端降队长、调 JVM 堆、加机器)都只是降低脆弱度,反馈环仍在。这类"温和亚稳态"未酿成对外事故,靠的是关键指标监控 + 及时响应。

## 受控复现实验(§5)

三个示例应用,验证模型并量化脆弱度:

1. **GC 亚稳态(Java/Docker,1GB 容器)**:每 job 分配 1MB 二维数组。注入 100% stall(trigger)制造积压。复现了队长↔GC↔应用停顿的互相关;脆弱度图(Figure 4c)显示 RPS 越高、堆越小(256MB vs 384MB),越短的 stall 就能推入亚稳态——但调大堆只是降脆弱度,不能免疫。
2. **Retry 亚稳态(MongoDB Raft 三副本 RSM)**:基线 6200 RPS,客户端 3s 超时最多重试 4 次;对 primary 容器限 CPU 作 capacity-decreasing trigger。结果极其敏感:**78% 降速 10s → 自愈;80% 降速 10s → 亚稳态**(尝试 RPS 冲到 2 万 = 3x,goodput 跌 ~90% 到 600 RPS,延迟顶死在 3s 超时);**80% 降速 9s → 自愈**;同 trigger 下基线降 30% 到 4200 RPS → 自愈。即 CPU 差 2%、时长差 1s 就是恢复与瘫痪的分界;trigger 结束前故障/retry 开始抬头,说明**及时撤掉 trigger 能阻止跃迁**。
3. **Look-aside cache 亚稳态(Nginx+PHP+MySQL 15GB+memcached 1GB,Zipf 分布)**:删热点 key 作 trigger。命中率掉 → DB 超时 → 应用来不及回填 cache → 命中率持续低。脆弱度随 RPS 升高而升;**超时 1s→2s 降低脆弱度**(换取更慢的故障发现);**稳态命中率 ~95% 比 ~80% 更抗跌**(Zipf 更倾斜,热点更容易回填),但高命中率支持更高 RPS,脆弱区间反而更宽。

## 讨论(§6)

- **多系统级联**:cache 的容量降级 = 存储系统的负载尖峰,组件间耦合使维持效应跨系统存活,即使单组件内部没有放大机制。
- **人为因素**:AZR4 在长假期前的周五低负载时仓促部署 bug 代码,部署流程因低负载没暴露容量下降,节后流量恢复即崩。
- **Fix to Break(越修越坏)**:不理解反馈环时的"修复"会放大脆弱性——AWS2(SimpleDB)事故后工程师把"重试锁服务失败后自我降级"改为**无限重试**,反而加重放大;一年后 AWS3(DynamoDB)同类事故重演。SPF1 后给错误路径加大量日志,SPF2 中这些日志抬高了每次 retry 的成本,加剧过载。
- **Autoscaling 不是银弹**:抬 Cnorm 也抬 Cstable,但 99% 命中率的 cache 丢失 = 100x 放大,扩容速度和成本都未必赶得上反馈环;有状态组件还不一定能扩。

## 与 HotOS '21 框架的关系

- 把"亚稳态故障普遍"从断言变成数据(21 起公开事故),并给出 trigger/放大/缓解的定量分布。
- 模型扩展一:trigger 不再默认是负载上升,**容量下降**(bug 部署、配置变更、硬件减速)同样触发——这把 fail-slow、scalability bug、配置/升级类故障统一收编为 trigger 类别。
- 模型扩展二:维持效应不止 workload amplification,新增 **capacity degradation amplification**(GC、leader election churn、sustained degradation),并统一为 αL/αC 两个放大系数。
- 模型扩展三:用实验否定"脆弱态是二元的",给出 Cstable 公式和亚稳态边界(Theorem 2/4),把 HotOS '21 的 hidden capacity 概念变成可计算的量(Cnorm/(w\*L·w\*C))。
- 复现了 HotOS '21 的 retry 与 look-aside cache 两个思想实验,证明其在真实软件栈(MongoDB、MySQL+memcached)上成立,且边界异常锋利。

## 启示

- **重试必须有界**。w\*L 无上界 = 数学上没有稳定区。vLLM/SGLang 推理服务前置网关和客户端 SDK 的重试次数、重试预算(retry budget)应按 Theorem 2 反推:期望稳定区容量 = 标称容量 / 最大放大倍数。AWS 把"自我降级"改成"无限重试"的教训说明:面向恢复的直觉修法恰恰可能拆掉稳定区。
- **容量下降型 trigger 在 LLM 推理中更隐蔽**:一次 kernel/配置更新让 prefill 吞吐降 20%(相当于 AZR4)、一张卡 fail-slow、CUDA graph 失效 fallback,都不会改变到达负载,但已把系统从稳定态推入脆弱态。变更灰度期若恰逢低峰(假期效应),容量下降可能完全不被发现——压测/容量回归要在峰值等效负载下做。
- **KV cache 是教科书式 look-aside 死结候选**:prefix cache 命中率骤降 → prefill 算力需求放大数倍 → TTFT 排队超时 → 请求被取消/重试,新请求又产生新前缀,旧前缀永远等不到被复用回填的机会。热点前缀(系统 prompt、few-shot 模板)对应论文里"删热点 key"的 trigger,且 Zipf 越斜恢复越容易但也越容易在更高 RPS 下运行——vLLM/SGLang 的 prefix cache 容量规划应以"cache 全失时纯 prefill 容量"(hidden capacity)为安全线,而不是以高命中率下的标称吞吐为准。
- **GC/后台任务型 capacity degradation 的对应物**:推理引擎里的 KV cache 碎片整理/块迁移、Mooncake 这类分布式 KV 池的 rebalance/数据搬迁、PD 分离集群的扩缩容重配置,都是"后台活动抢占前台资源 → 队列更长 → 后台活动更多"的候选反馈环。Twitter 案例的教训:即使不能消除反馈环,量化其特征(GC 时长 vs 队长曲线)也能指导参数调优;且"负载已降但指标不恢复"本身就是亚稳态的识别信号。
- **亚稳态边界给出可操作的告警阈值**:监控"过载量"(排队积压的 RU,如 prefill token 积压)而不只是负载;当过载量逼近 α·mtrig 边界(即撤掉 trigger 也救不回来)前就应激进降载。对 PD 分离集群,trigger 期间(decode 节点重启、网络分区)每多拖 1 秒,w(Δttrig) 就增大一截——自动化的快速 trigger 切除(秒级)比人工响应价值大得多。
- **timeout 的双刃剑效应有实验依据**:cache 实验里超时 1s→2s 降低脆弱度;MongoDB 实验里 3s 超时决定 retry 放大启动延迟。推理服务的 TTFT/TPOT 超时和重试策略应放在"放大延迟"框架下权衡,而不是单纯按尾延迟目标拍脑袋。
