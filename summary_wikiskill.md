# WikiSkill: Compiling Agent Experience into Persistent Knowledge for Skill Evolution 总结

- arXiv: 2608.27454(Google Research,2026-08-28;作者:Liyan Tang, Cyrus Rashtchian, Chun-Sung Ferng, Andrew Tomkins, Da-Cheng Juan, Tu Vu),TeX 源码缓存于 `~/.cache/nanochat/knowledge/2608.27454/`(单文件入口 `main.tex`)。
- 与 kimi-code 的关联:WikiSkill 里的 "skill" 正是 Anthropic/kimi-code 采用的 SKILL.md 文件系统格式(`~/source_code/kimi-code/packages/agent-core/src/services/skill/skillService.ts` 管理 skill 的列举/激活;本机的 `~/.kimi-code/skills/read-arxiv-paper/SKILL.md` 就是一个实例),论文研究的是如何**自动演化**这类 skill。

**一句话**:WikiSkill 在 agent 工作区中引入三层知识架构——Raw(不可变执行轨迹)/ Wiki(持久结构化知识,永不回滚)/ Skills(可演化的过程性 skill),通过"推理 rollout → Wiki Maintainer 沉淀模式 → Skill Proposer 基于 wiki 提 skill 更新 → 验证集 gating 接受/回滚(仅回滚 skill,不回滚 wiki)"的循环持续演化 skill。在 5 个 benchmark、5 个模型上稳定优于 Trace2Skill/EvoSkill/SkillOpt 等 skill 演化基线;消融证明持久 wiki 是关键(去掉后平均分 63.7% → 48.7%)。

---

## 1. 动机与问题设定(§1–2)

- **Agent skill** = 以文件系统目录形式打包的可复用过程性知识(指令 + 脚本 + 资源),核心是带 frontmatter(name + description)的 `SKILL.md`。它不改模型参数、支持 progressive disclosure(按需加载,省 context),但大多靠手写。
- 已有的自动 skill 演化方法(EvoSkill、Trace2Skill、SkillOpt)都把"学到的东西"散落在优化历史/反馈记录里,**没有一份独立、持续演化的知识表示**。受 Karpathy 的 "LLM Wiki" 观点(把经验编译成持久、可复利的知识)启发,作者问:agent 经验能否也被编译成持久知识来支撑长期 skill 演化?
- 形式化:系统状态为 $(S_k, W_k)$——skill 集 + Wiki;从 $(\emptyset, \emptyset)$ 出发,训练 rollout 产生经验,gating 基于验证集,目标是最大化测试集表现。skill 更新可回滚,wiki 永不回滚。

## 2. 方法(§3)

### 2.1 三层知识架构(§3.1)
- **Raw Layer(`raw/`)**:不可变的完整执行轨迹(推理、工具调用、工具输出、最终答案),供 Wiki Maintainer 和 Skill Proposer 事后分析。
- **Wiki Layer(`wiki/`)**:持久知识库,跨迭代累积,包含:
  - `patterns/`:每个 markdown 一页,记录失败模式/成功策略 + 根因 + 可执行 workaround;
  - `index.md`:模式目录(每条必须写清 问题 + 根因 + 修复);
  - `logs.md`:演化日志(Wiki Maintainer 更新);
  - `skill-impact.md`:skill 影响追踪器(由外层 harness 在 gating 后**程序化**写入:proposal 元数据、unified diff、验证分数、接受/拒绝结果)——构成客观的审计轨迹,防止重复提议被拒绝过的方案。
- **Skill Layer(`skills/`)**:当前生效的 skill 集。每个 skill 目录含 `SKILL.md`(完整内容)和 `PURPOSE.md`(记录该 skill 由哪些 wiki pattern 催生——skill 到知识的溯源映射)。

### 2.2 演化循环(§3.2)
每轮迭代 4 个组件:
1. **Inference Agent**:用当前 skill 集在训练集上 rollout。**skill 全文直接注入 system prompt**(而非按需检索),以排除 skill 触发/检索失败这一混淆变量;训练 rollout 期间**禁止访问 wiki**(消融证明允许访问反而降低最终 skill 质量)。
2. **Wiki Maintainer**:分层采样 ≤8 条轨迹(≤5 失败 + ≤3 成功,单条截断至 15K 字符),做根因分析,以增量 patch(append/replace/insert_after)更新 wiki 模式页、索引和日志。创建/更新模式数无硬上限。
3. **Skill Proposer**:多轮 ReAct agent(约 10–20 轮),初始只给 wiki 索引、skill-impact.md 和训练任务结果摘要,通过 `read_file` 按需翻模式页和原始轨迹,诊断根因后产出**原子化 proposal**(每次只针对一个 skill:新建或 patch 编辑);规则要求至少读 4 条轨迹、优先 patch 而非新建、不得重复被拒绝的方案。
4. **Gating & Rollback**:候选 skill 集在验证集上评估,分数超过历史最优才接受,否则回滚 skill;**无论接受与否,wiki 都保留**。验证集满分时提前终止。之后 harness 程序化追加 `skill-impact.md`。

### 2.3 API 调用复杂度(附录 C)
Wiki Maintainer 每轮 1 次调用 + Proposer 的 ReAct 轮数;全批量(B = N_train)下每轮优化器调用数与训练集大小无关(O(1)),而 EvoSkill/SkillOpt 随 N_train/B 线性增长,Trace2Skill 至少 O(N_train)(每条轨迹独立分析)。

## 3. 实验结果(§4)

- **设置**:5 个 benchmark——LiveMath(数学竞赛推理)、SealQA(网页搜索问答)、SpreadsheetBench(表格操作,bash 工具)、OfficeQA(长文档 QA,grep/read 工具)、ALFWorld(具身交互);5 个模型——Gemini-3.5-Flash、Qwen-3.5-4B/9B、Qwen-3.6-27B、Gemma-4-31B;全部 3 次独立完整演化取平均,paired bootstrap 显著性检验(p<0.05)。
- **主结果(Table 1)**:WikiSkill 在全部 5 个模型上平均分第一,相比每个模型最强基线分别高 3.3 / 5.1 / 10.0 / 5.8 / 12.0 分。典型例子:Gemini-3.5-Flash LiveMath 33.0→72.6,SpreadsheetBench 50.5→76.6;Qwen-3.6-27B ALFWorld 52.8→77.6。基线方法不稳定(如 EvoSkill 在 Qwen-9B LiveMath 大涨但在 Gemma-31B 上反降)。
- **skill 演化与模型规模互补**:Qwen 家族内平均提升随规模增大(4B +12.3 → 9B +17.5 → 27B +23.9);同时小模型 + skill 可超过大模型裸跑:Qwen-3.5-9B + WikiSkill 47.4% > Qwen-3.6-27B 无 skill 39.4%。
- **跨模型迁移(Table 2)**:skill 可跨规模、跨家族迁移,且**他模型演化的 skill 常优于自演化**(如 Qwen-3.5-9B 用 Qwen-3.6-27B 的 skill 在 ALFWorld 达 70.2% vs 自演化 63.4%;甚至 4B 演化的 LiveMath skill 让 Gemma-31B 从 33.9→73.1)。说明"发现过程性知识"与"执行过程性知识"是两种不同能力。但也有**负迁移**:Qwen-4B 的 SpreadSheet skill(单行 Python、字符串转换等低层 workaround)让 Gemini-3.5-Flash 从 50.5 掉到 18.1——skill 若编码了特定模型的规避技巧而非通用流程,会束缚更强模型;且碎片化诊断流程会耗尽交互预算。迁移效果还取决于目标模型的执行能力(4B 在 OfficeQA 长上下文下跟不上多步检索 skill,反而退化)。
- **数据、benchmark 详情**:各 benchmark train/val/test 划分与工具配置见附录 B,与 SkillOpt/EvoSkill 严格对齐;验证集较小(10–40 条),靠 3 次重复实验降噪。

## 4. 消融与分析(§5)

- **wiki 是核心贡献(Gemini-3.5-Flash 消融,Table 3)**:默认配置(推理 agent 不看 wiki、Proposer 看 wiki)平均 63.7%。Proposer 也拿不到 wiki(同时移除 Maintainer)= 48.7%(−15.0);推理 agent 训练时也看 wiki = 60.9%(−2.8)——假设是推理 agent 直接从 wiki 拿知识完成任务,产生的轨迹对 skill 开发信息量下降。
- **wiki 持续累积、skill 保持精简**:平均每个设置创建 6.3–8.9 个模式页、编辑 7.0–18.4 次;skill 平均 45–129 行(Qwen 系偏长,Gemini/Gemma 紧凑)。
- **演化全程都在改进**:被接受的 skill 更新只有 39%–58% 发生在前两轮,中后期仍大量接受(SealQA 后期占 28%)——持久知识支撑了持续精化。
- **案例(Qwen-3.6-27B @ ALFWorld)**:第 0 轮 `goal-directed-action` 被拒,skill-impact.md 记下 diff;第 1 轮据此新建 `break-repetition-loop`("Never Return an Item to Its Origin Location")被接受;第 4 轮 wiki 累积新循环变体后进一步 patch 出"Each Operation Type ONCE Per Item"。展示了被拒方案的历史记录如何指导后续演化。

## 5. 局限

1. skill 全文注入 prompt,未评估 skill 检索/触发(规模大了之后重要);
2. 严格 gating(必须提升验证分)排除了"中性但为后续增益铺路"的 proposal;
3. wiki 只增不删,缺少自动剪枝机制;
4. 未覆盖数百步/数小时级的超长 horizon 任务(在线单轨迹内 skill 适应是未来方向)。

---

## 6. 与本工作区(kimi-code / skills)的联系与启发

这篇论文与 `~/source_code/kimi-code/` 的关联比一般推理引擎论文更直接——它研究的正是 kimi-code 所采用的 SKILL.md 生态如何自动化演化:

- **格式同源**:论文的 skill 定义(SKILL.md + frontmatter name/description + progressive disclosure)与 kimi-code 的 skill 系统一致(`packages/agent-core/src/services/skill/` 下 skillService/skill.ts 负责 list/activate;协议层 `@moonshot-ai/protocol` 的 `SkillDescriptor`)。kimi-code 目前 skill 全靠手写/人工安装,WikiSkill 给出了从会话轨迹自动生成和迭代 skill 的完整范式。
- **可借鉴的设计**:
  1. **三层分离**:kimi-code 有会话 transcript(`packages/transcript/`)但没有"把轨迹沉淀成持久知识库"的中间层;WikiSkill 的 wiki(patterns + index + logs + skill-impact)可视为 AGENTS.md/记忆文件的结构化升级版。
  2. **PURPOSE.md 溯源**:skill 与其催生的知识/教训显式互链,值得借鉴到 skill 元数据设计。
  3. **skill-impact.md 审计轨迹**:由 harness 程序化(而非 LLM)记录每次改动 diff 和验证结果,防止"重复犯同一个错"——对任何自动 prompt/skill 优化循环都适用。
  4. **gating 只回滚 skill、不回滚知识**:知识累积是单调的,即使某次 skill 改动错了,教训也留下。
  5. **反面教训**:演化期让执行 agent 直接读 wiki 会污染轨迹、降低 skill 质量——知识应该"编译进 skill"而不是让执行时绕过去查。
- **对评估/选用 skill 的启发**:skill 的收益随模型能力增长(强模型更能执行复杂流程),且小模型的 skill 可能编码只对自己有用的 workaround——跨模型搬 skill(比如把为某模型写的 skill 库直接给 Kimi K3 用)未必总是正收益,值得实测。
