# Metastable Failures 研究线汇总(6 篇论文综合)

本文汇总 HotOS'21《Metastable Failures in Distributed Systems》及其相关工作的 6 篇独立解读,梳理这条研究线的演进脉络、核心共识与分歧、以及对 LLM 推理系统的综合启示。

独立文档(均在 ~/knowledge/ 下):

| # | 文档 | 论文 | 定位 |
|---|------|------|------|
| 0 | summary_metastable_failures.md | HotOS'21, Bronson 等 | 问题定义(vision) |
| 1 | summary_metastable_failures_wild.md | OSDI'22, Huang 等 | 野外实证 + 负载/容量模型 |
| 2 | summary_analyzing_metastable.md | HotOS'25, Isaacs 等 | 预测工具链(CTMC→DES→emulator→压测) |
| 3 | summary_characterizing_metastable_faults.md | arXiv 2026, Farahbakhsh 等 | 因果重刻画(metastable fault + MFT) |
| 4 | summary_breakwater.md | OSDI'20, Cho 等 | 机制侧:过载控制样板 |
| 5 | summary_gray_failure.md | HotOS'17, Huang 等 | 概念近亲:检测维度的互补 |

## 一、演进脉络:定义 → 实证 → 预测 → 因果

这条线的四篇主线论文恰好构成一个完整的研究循环:

1. **定义 (HotOS'21)**:提出三态模型(stable/vulnerable/metastable)、trigger vs sustaining effect、hidden capacity、characteristic metric 等概念。贡献是"给现象命名",方法是工业经验归纳,结论是"系统性防御是开放问题"。
2. **实证 (OSDI'22)**:用 21 起公开事故验证框架的普遍性,并做了两个关键扩展:**trigger 不止负载尖峰,容量下降(bug 部署、配置变更)同样触发;维持效应不止 workload amplification,还有 capacity degradation amplification**(GC 风暴、leader election churn)。给出 Cstable = Cnorm/(w*L·w*C) 的量化边界——retry 不封顶则数学上不存在稳定区。实验最震撼的发现是**脆弱边界的锋利性**:MongoDB 上 78% CPU 降速 10s 能自愈,80% 降 10s 即瘫痪,80% 降 9s 又自愈——2% CPU / 1 秒之差决定生死。
3. **预测 (HotOS'25)**:回答"出事之前能不能算出脆弱区"。用 CTMC(连续时间马尔可夫链)给亚稳态下数学定义(两个互达状态子集间最小期望到达时间远超自然时间尺度),毫秒级生成"队列长度 × 在轨重试数"向量场扫描配置空间,再逐层升级到离散事件仿真、生产框架 emulator、真实压测,层间对账校准。核心工程结论:对运维最有用的交付物是**"一个能按需翻倒系统的配置"**。并强调:排队论意义下稳定的系统仍可亚稳——稳定 ≠ 安全。
4. **因果 (arXiv 2026)**:批评前三篇是"现象学"——只从症状(高延迟、低 goodput)出发,导致补救永远只会"卸负载"。提出根因是结构性的 **metastable fault**:各组件都在自我稳定、但其稳定化动作恰好互相去稳定化所形成的环("组合之罪")。Theorem 1:fault 是 failure 的必要条件(无环即可排除亚稳态);fault 是否点燃取决于**调度**——由此导出两条可操作的设计原则:R1(去稳定化动作试探性、可推迟)+ R2(稳定化动作优先直到全局稳定)。配套 MFT 设计方法论和 Nyx DSL 工具。

一句话概括演进:**从"它长什么样"(现象)→ "它有多普遍"(实证)→ "能不能提前算出来"(预测)→ "它的因果结构是什么"(机理)**。

## 二、横向对比

**根因观的分歧(最主要的学术争论)**:
- HotOS'21/OSDI'22:根因 = sustaining effect(反馈环),应对重心是削弱环、卸负载、过载时切策略。
- arXiv 2026:根因 = metastable fault(互相去稳定化的动作环),卸负载只是治标;真正的杠杆在**调度策略**(R1/R2),能消环时直接消环。两者不矛盾:fault 是结构,sustaining effect 是该结构在运行时的动力学表现;2026 论文的贡献是让"找环"从事后艺术变成可证明的设计纪律。

**两条互补的故障分析维度**:
- **Gray failure(HotOS'17)回答"怎么看见"**:故障本质是 differential observability——app 已受害而系统内部 observer 认为健康。防御靠多维度监控、逼近 app 视角的探测、独立观测平面。
- **Metastable 回答"为什么停不下来"**:动力学/反馈环维度。
- 二者可以复合:Azure Storage 案例中,容量上报 bug 让 manager 看不见 gray failure → 持续写入 → 崩溃 → 摘除 → 压力传染——**灰色故障喂养亚稳态反馈环**。metastable 的 characteristic metric 本质上就是 gray failure 意义上"逼近 app 观测"的指标选择。

**机制侧的正面样板(Breakwater, OSDI'20)**:
- 与 metastable 论文互相印证的关键结论:**排队延迟是唯一可靠的过载信号**(CPU 利用率分不清健康满载和 livelock,队列长度扛不住服务时间长尾),既鲁棒又可直接映射 SLO。
- 证明"把系统挡在脆弱阈值之外"在工程上可行:信用准入 + AIMD 调整 + 超量授信 + 延迟驱动 AQM,需求突跳 1.4× 容量时 <20ms 收敛,比 DAGOR/SEDA 快一个数量级。
- 局限同样说明问题:只覆盖单机单层,跨层过载传播(work amplification 传染)仍然开放。

## 三、贯穿六篇的核心洞见

1. **效率和可靠性 feature 是故障温床**。重试、failover、cache、优化、甚至冗余(gray failure 的 Clos 网络案例:fan-out 越大,任一交换机随机丢包拖慢几乎所有请求)——稳态收益越大,往往把系统推得离脆弱边界越近。
2. **稳定 ≠ 安全,健康 ≠ 不脆弱**。系统可以在脆弱态正常运行数月数年;排队论稳定的系统仍可亚稳;observer 健康 ≠ app 健康。
3. **边界锋利且不可见**。2% CPU、1 秒之差决定自愈还是瘫痪;hidden capacity 由故障态行为决定,正常运营中测不到——这是容量规划最大的盲区。
4. **指标选择是胜负手**:排队延迟类指标 > 速率类指标(QPS);能逼近 app 视角、反映反馈环"记忆"的 characteristic metric 是告警、压测、模型校准三者的共同抓手。
5. **应对分四层,缺一不可**:消环/设计纪律(R1/R2、read-through cache、retry budget)→ 准入控制挡在阈值外(Breakwater)→ 检测(Gray failure 的多视角观测 + characteristic metric)→ 恢复(压测工具链提前备好"排空集群"的能力,Kraken 式引流)。
6. **组织维度被反复点名**:"fix to break"(AWS SimpleDB 事故改成无限重试,一年后 DynamoDB 重演);奖励稳态优化 = 奖励靠近脆弱边界。技术问题有技术解,激励问题没有。

## 四、对 LLM 推理系统的综合启示(vLLM/SGLang/Mooncake/KTransformers)

- **KV cache 是 look-aside 死结的头号候选**:prefix cache 命中率骤降(重启、换模型、流量模式切换)→ 每请求算力需求倍增 → 排队延迟暴涨 → 超时重试 → 缓存进一步失效。容量安全线应取 **cache 全失时的 hidden capacity**,而非命中时的标称吞吐。
- **容量下降型 trigger 在推理系统里更隐蔽**:kernel 更新、量化配置变更、新模型部署造成的吞吐下降,都是 OSDI'22 定义的 capacity-decreasing trigger,且不表现为任何"故障"。
- **重试纪律**:retry 不封顶则数学上没有稳定区(OSDI'22 定理);推理网关应实施 retry budget + 退避(R1 原则),重试流量进独立低优先级队列(请求/重试分队列是 R2 的实例)。
- **过载信号用排队延迟**:prefill 队列排队时间/TTFT 排队分量是比 QPS、GPU 利用率更鲁棒的 characteristic metric——与 Breakwater、HotOS'21 的结论三方印证。
- **PD 分离与弹性扩缩容的双重风险**:扩缩容的冷启动(KV cache 预热、权重加载)期间有效容量反而下降,可能在脆弱态下成为容量下降型 trigger;Mooncake 式 KV 池化降低了单点 cache 丢失的影响,但池本身的过载传播需要准入控制。
- **观测平面**:推理系统的 gray failure 形态——健康检查(进程存活、GPU 可见)全绿而 goodput 已崩——需要逼近用户视角的 canary(真实推理请求探针)和独立于推理栈的观测聚合。

## 五、推荐阅读路径与引用关系

按"现象 → 实证 → 预测 → 因果"读主线四篇(0→1→2→3),机制与检测两篇(Breakwater、Gray failure)可在任意时点插入;Breakwater 适合在读完 HotOS'21 §3 应对手段后读,作为"防御能做到什么程度"的正面参照;Gray failure 适合最后读,把视角从动力学拉回观测,形成完整闭环。

引用关系:OSDI'22、HotOS'25、arXiv 2026 均以 HotOS'21 为起点;Breakwater 与 HotOS'21 同期独立、互相印证;Gray failure 先于 HotOS'21,是其"无 bug outage"家族的先声。
