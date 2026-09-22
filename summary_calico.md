# Calico: Making Array-Based Translation Practical for Modern, High-Performance Buffer Management

- arXiv: 2604.00423(VLDB 投稿模板)
- 作者: Xinjing Zhou (MIT CSAIL), Jinming Hu (sea-land.ai), Andrew Pavlo (CMU), Michael Stonebraker (MIT)
- 原文缓存: `~/.cache/nanochat/knowledge/2604.00423/`(TeX 源码)

## 一句话总结

用「数组直查」(page ID 直接当下标)替代哈希表做 buffer pool 的页号→内存帧映射,配合多级稀疏数组 + hole punching + group prefetch,在保持 DBMS 完全控制驱逐/I/O 的前提下,达到甚至超过 mmap/vmcache(OS 页表翻译)的性能;PostgreSQL 中 2.6K 行改动替换 buffer manager,pgvector 向量检索最高加速 6.5×。

## 背景与动机

现代 buffer pool 不再只服务 OLTP(B-tree 点查),还要服务:
- **顺序扫描(SS)**:heap scan、IVFFlat 等分区型向量索引 → 考验空间局部性与硬件预取;
- **范围扫描(RS)**:先一次依赖查找再顺序扫(B-tree 叶子链);
- **点查(PL)**:B-tree 自顶向下,每次访问都是依赖加载,延迟敏感;
- **图遍历(GT)**:HNSW 向量检索,高扇出不规则访问 → 考验 memory-level parallelism(MLP)。

现有设计家族各只能满足一部分需求:
- **哈希表**(PostgreSQL/MySQL 等主流):用户态、DBMS 完全控制,但哈希计算 + 指针追逐 + 同步在关键路径上;哈希打散相邻 PID,破坏局部性;高并发下原子操作 coherence 流量大。
- **OS 页表翻译**(mmap、vmcache):硬件加速翻译快,但 DBMS 失去驱逐/I/O 控制;根本矛盾是 **eviction granularity mismatch**——buffer pool 要 4KB 细粒度驱逐,TLB 效率要 2MB 大页,而 OS 无法只驱逐 2MB 大页里的一张 4KB 页(只能整页驱逐 / 拆大页 / 拒绝),于是 mmap 系被迫二选一。vmcache 用 4KB 页;vmcache+exmap / Tabby+libdbos 需要改内核,部署受限。
- **Pointer swizzling**(LeanStore/Umbra):热路径零翻译开销,但要侵入数据结构内部,对多父引用(图索引 fan-in、兄弟链、二级索引)很难做。
- **Predictive translation**(PrediCache):给页分配偏好帧位置让 CPU 推测执行,workload 依赖强。

**核心观察**:1984 年 Effelsberg & Haerder 就讨论过 array translation,但因为真实 DBMS 的 PID 是大而稀疏的层级 ID(32–128 bit),平铺数组的虚拟内存/物理内存开销不可行,被放弃至今。本文证明:现代硬件 + 正确的空间管理技巧下,这个设计点成立。

## 翻译性能微基准(Sec. 3,单线程、全驻留内存)

统一接口:`page_id (8B) -> frame 指针`,对比 std::unordered / absl::flat_hash / predicache / calico array / vmcache(4KB 与 2MB):

| 负载 | 结果 |
|---|---|
| 顺序扫描 | array 比 chained hash **快 6.9×**,与 vmcache(2MB) 持平;LLC miss 1.33/页 vs hash 2.53 |
| 随机范围扫描 | array 仍快 2.0× |
| B-tree 点查 | array 0.93 Mops/s,与 predicache 持平(但指令更少、LLC miss 更低:0.46 vs 1.30);vmcache(2MB) 略快(1.03)但失去细粒度驱逐 |
| 图 BFS(5M 节点,44 邻居) | array 15.5s,**比 absl hash 快 2.7×**,还快过 vmcache(4KB);hash 系 IPC 仅 0.18–0.22,数组 0.63 |

两个叠加的原因:(1) 无哈希/指针追逐/锁,消除串行化数据依赖,让 CPU 并行发出多个加载;(2) 数组项只存 8B frame ID(PID 隐含在下标里),密度远高于哈希表(键值对 + load factor 浪费),缓存更省。

## Calico 设计

架构上把 **逻辑翻译(DBMS 管)** 与 **物理帧内存(OS 管)** 解耦:

1. **多级翻译 + path caching**:PID 拆成 prefix(如 PostgreSQL BufferTag 的 Tablespace/DB/Relation/Fork)+ suffix(BlockNumber);上层索引用哈希表把 prefix 映射到最后一级翻译数组,后缀直接数组下标。利用 DBMS 访问天然的 prefix 局部性,每线程 TLS 缓存最近的 (prefix → 数组指针),命中时热路径只有一次数组访问(pgvector 场景命中率接近 100%)。
2. **64-bit TranslationEntry 打包**:frame ID(32b)+ version(24b,乐观读校验)+ latch state(8b),一次原子加载同时拿到翻译与并发元数据。**全 0 = evicted** 的不变量是空间优化的关键。
3. **按需物理内存**:翻译数组用 `mmap` 预留巨大虚拟空间,初始映射到共享零页(读缺页不耗物理内存,写才 COW 分配)。配合全 0=evicted 编码,驱逐时把 entry 写回 0。
4. **Active hole punching(HPArray)**:每个 OS 页(512 个 entry)一个 32-bit 引用计数,某组 entry 全部驱逐后对该页 `madvise(MADV_DONTNEED)` 把物理内存还给 OS。加锁顺序保证并发安全(驱逐先拿 metadata 锁再释放 entry 锁,page fault 先增计数再发布 frame ID)。比 OS 页表还强:OS 换出页留下非零 swap entry 无法回收页表,Calico 显式清零可回收。
   - 实测:TPC-C(16GB pool)回收 86.2%,YCSB-D(read-latest)回收 95.7%,翻译内存最终 68MB,**只有哈希表的 0.5×、vmcache 的 1/47**;最差情况(YCSB-C zipf 热点散布)退化为与哈希表相当,仍比 vmcache 省 2×。
5. **Group prefetch**:面向 HNSW 高扇出暴露 MLP——先批量 prefetch 翻译项,再 prefetch 驻留页的指定偏移,非驻留页批量提交异步 I/O。关键:只有数组翻译吃到了收益(1.201×),hash/predicache 的 prefetch 本身还要做哈希探测,vmcache 反而回退。
6. **乐观读接口**:snapshot entry → 读 frame → 校验 version/frameId 未变,免 pin/unpin 原子操作;失败回退标准 pin/unpin。

## 评估结果(双路 EPYC 7513 服务器)

- **向量检索(HNSW,64 线程)**:内存内与 vmcache 打平、超 USearch;超内存时比 vmcache 快 2×、比 USearch(mmap 同步缺页)快 6×——vmcache 的差距来自 `madvise(DONTNEED)` 驱逐引发的全核 TLB shootdown,Calico 用户态驱逐完全避开。
- **OLTP**:内存内 YCSB-C 与 LeanStore/vmcache 相当(单线程 LeanStore 仅快 9%),TPC-C 128 线程与 LeanStore 持平;超内存时 YCSB-C 比 vmcache/LeanStore 快 1.6×(per-core 优势明显,对手把 CPU 烧在 TLB shootdown 上)。
- **PostgreSQL v18 + pgvector**(2.6K 行 buffer manager 改动 + 600 行 pgvector 改动):
  - HNSW 内存内:数组翻译 1.32–1.52×,加乐观读 2.5–2.77×,加 group prefetch **3.84–3.95×**,追平内存库 Faiss;
  - 2GB pool(超内存):group prefetch **5.83–6.57×**;
  - 顺序扫描 SUM:随行数增大 1.17×(128B 行)→ 2.99×(6KB 行),几何均值 1.70×;IVFFlat 查询延迟降 1.38×。
- **Ablation**(DEEP10M HNSW):数组 1.59× → 大页帧 +1.35× → 乐观读 → group prefetch,累计 3.15×。
- **跨平台**:ARM Cortex-A72 与 Intel i9-13900K 上结论一致(ARM 对大页/TLB 更敏感,gap 更大)。

## 对本工作区(LLM 推理基础设施)的联想

这篇是数据库论文,但思想与推理系统的存储/缓存层高度同构:

1. **「数组直查 vs 哈希表」与 KV cache block 管理同构**。vLLM 的 `BlockManager`/prefix cache 用哈希表(block hash → physical block),PG 的 page→frame 翻译同理。如果 KV block 的虚拟 ID 空间可以设计得稠密可索引,数组直查 + 硬件预取的思路对 scan 型访问(如长上下文逐 block 加载、RadixAttention 复用检查)有借鉴价值;前提是 ID 分配器要产出局部性好的 ID。
2. **eviction granularity mismatch 在 GPU/异构内存侧同样存在**:CPU 侧 2MB 大页 vs 4KB 细粒度驱逐的矛盾,对应 GPU 统一内存/CPU offload 里 page 粒度的迁移开销;「逻辑翻译与物理后备解耦」(Calico 的核心)就是把迁移粒度与访问粒度分开的通用解法。
3. **TLB shootdown 是 mmap 路线在驱逐频繁时的隐性税**:vmcache 超内存时输 2× 的原因值得记住——任何用 `madvise`/缺页交给 OS 的设计,在内存压力下都要付全核 shootdown;推理 KV cache 若走 mmap 式 host 内存管理同理。
4. **Hole punching 的「全 0 = 空」不变量**是稀疏数组实用化的精髓:虚拟空间大胆开大,物理内存靠零页共享 + 引用计数回收,物理足迹只跟工作集走。对超大稀疏 KV cache 元数据(如跨请求 prefix cache 的全局目录)有参考价值。
5. **Group prefetch 只对「翻译路径本身是 O(1) 数组访问」的设计有效**——哈希/predicache 上 prefetch 收益被翻译开销吃掉。这提示:先简化元数据访问路径,再谈预取/推测,顺序不能反。

## 备注

- 论文自评 worst case:病态稀疏下(512 页组里只有 1 页驻留)每组 4KB 元数据开销,与 OS 页表同级;提出的 adaptive hybrid(低密度区域迁移到后备哈希表)留作 future work。
- 代码实现为 PostgreSQL drop-in buffer manager(v18)+ pgvector 扩展修改;实验 baseline 包括 LeanStore(629b41a)、LMDB 0.9.31、WiredTiger 10.0.2、vmcache、USearch、absl::flat_hash、PrediCache。
