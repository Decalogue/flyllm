# 向量数据库选型：Milvus vs Qdrant

## 1. 核心定性

**Milvus** 本质上是一个面向**海量十亿级向量**的**云原生分布式**向量数据库，通过**存储计算分离 + 分片分区分桶**架构实现高吞吐高可用。

**Qdrant** 本质上是一个面向**中小规模高性能检索**的**单体式**向量数据库，通过**HNSW 索引 + 内存映射 + 过滤器下推**实现毫秒级延迟。

---

## 2. 主流程对比

| 维度 | Milvus | Qdrant |
|------|--------|--------|
| **写入流程** | 数据 → Proxy 负载均衡 → DataNode 分片 → WAL 持久化 → 异步构建 IVF/HNSW 索引 | 数据 → 内存索引更新 → WAL 追加 → 异步刷盘 |
| **检索流程** | QueryNode 并行检索分片 → Reduce 聚合 TopK → Proxy 返回 | HNSW 图遍历 + payload 过滤器下推 → 直接返回 |
| **扩容方式** | 水平扩展（加 QueryNode/DataNode） | 垂直扩展（加 CPU/内存）或副本集 |

---

## 3. 数学基础

**向量相似度计算核心公式**（两者通用）：

$$\text{cosine\_similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\sum_{i=1}^{d} a_i b_i}{\sqrt{\sum_{i=1}^{d} a_i^2} \sqrt{\sum_{i=1}^{d} b_i^2}}$$

**Milvus IVF 索引检索复杂度**：

$$T_{search} = O\left(\frac{n}{nlist} \cdot d\right) + O(nprobe \cdot \frac{n}{nlist} \cdot d) \approx O(nprobe \cdot \frac{n}{nlist} \cdot d)$$

其中：
- $n$: 向量总数
- $nlist$: 倒排列表数量（聚类中心数）
- $nprobe$: 查询时扫描的聚类中心数
- $d$: 向量维度

**Qdrant HNSW 索引检索复杂度**：

$$T_{search} = O(m_{max} \cdot \log n \cdot d)$$

其中：
- $m_{max}$: 图的最大出度（默认 64）
- $n$: 节点总数
- $d$: 向量维度

---

## 4. 工程考量

### Trade-offs

| 维度 | Milvus | Qdrant |
|------|--------|--------|
| **一致性模型** | 最终一致性（Gossip 协议传播元数据） | 强一致性（单节点 Raft/Leader-Follower） |
| **内存策略** | 磁盘为主，热数据缓存（受 `queryNode.cacheSize` 控制） | 内存为主，mmap 冷数据 fallback |
| **索引构建** | 异步后台构建，支持 GPU 加速 | 实时增量更新，无延迟尖刺 |

### 致命弱点

**Milvus**:
- **冷启动灾难**: 十亿级数据首次加载时，若 cache 未命中，磁盘随机读将导致查询延迟从 ms 级退化到秒级
- **元数据膨胀**: Collection 过多时，Etcd 中的元数据量激增，RootCoord 可能成为瓶颈
- **脑裂风险**: 多 Proxy + 多 QueryNode 架构下，网络分区时可能出现短暂双主

**Qdrant**:
- **单机天花板**: 受限于单机内存，超过 ~5000万 768维向量（约 150GB）后性能断崖下跌
- **写入放大**: HNSW 图结构更新需要重建多层连接，高并发写入时 CPU 100% 争抢
- **过滤器失效**: 复杂 payload 过滤条件无法下推时，退化为暴力扫描

---

## 5. 工业映射

**Milvus** 被应用于：
- **OpenAI**: Embedding Store 的底层存储（十亿级向量召回）
- **Shutterstock**: 亿级图片相似搜索
- **场景**: 需要**水平扩展 + 多租户隔离 + 复杂过滤**的企业级 RAG 系统

**Qdrant** 被应用于：
- **Cohere**: 轻量级 Embedding 缓存服务
- **Huggingface**: 模型语义搜索 Demo
- **场景**: 追求**极致延迟 + 快速部署 + 中等数据量**的 AI 应用原型

---

## 选型决策树

```
数据规模？
├── < 1000万 → Qdrant（部署简单，延迟低）
├── 1000万 ~ 1亿 → 看团队规模
│   ├── 有运维团队 → Milvus（预留扩展空间）
│   └── 小团队/快速上线 → Qdrant
└── > 1亿 → Milvus（唯一选择）
```

**一句话总结**: 要**扩展性**选 Milvus，要**低延迟**选 Qdrant。
