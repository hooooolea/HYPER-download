# HyperGraphRAG 项目学习详细总结

> 本文档面向零基础用户，通过一个数学文档的完整例子，详细说明 HyperGraphRAG（超图知识图谱检索增强生成）的概念、架构、数据流和代码实现。

---

## 一、核心概念：什么是 HyperGraphRAG？

### 1.1 从普通 RAG 说起

**RAG（检索增强生成）**的工作流程：
```
用户提问 → 检索相关文档 → 将文档内容发给 LLM → LLM 生成回答
```

普通 RAG 的问题：**它把文档当作一堆文本块来处理，不理解内容之间的关联**。

### 1.2 HyperGraphRAG 的改进

HyperGraphRAG 在普通 RAG 基础上增加了**知识图谱层**，核心思想：

> 把文档内容抽象成**实体**（如"函数"、"单调递增"）和**关系**（如"包含"、"依赖"），构成一张图。然后通过图的结构来检索，而不是简单的文本相似度。

```
普通 RAG：    文档块 → 文本向量 → 相似度检索
HyperGraphRAG：文档块 → 实体+关系图 → 图遍历检索
```

### 1.3 什么是"超图"（HyperGraph）

普通图的边只能连接两个节点。超图的边（称为 **hyperedge**）可以连接任意数量的节点。

在 HyperGraphRAG 中：
- **节点** = 实体概念（函数、单调递增、奇偶性……）
- **超边** = 关系，可以有两个或多个参与者

---

## 二、系统架构：六类存储组件

HyperGraphRAG 用到 **6 种存储**，各司其职：

### 2.1 存储一览表

| 组件名 | 类型 | 存储内容 | 数据规模 |
|--------|------|----------|----------|
| `full_docs` | KV 存储 | 原始文档全文 | 1 篇文档 |
| `text_chunks` | KV 存储 | 分块后的文本（含章节、领域信息） | 3 个 chunk |
| `chunk_entity_relation_graph` | 图存储 | 实体关系网络（NetworkX 实现） | 6 节点，11 边 |
| `concept_vdb` | 向量存储 | 实体概念的语义向量 | 3 个向量 |
| `relations_vdb` | 向量存储 | 关系（边）的语义向量 | 11 个向量 |
| `fragment_vdb` | 向量存储 | 文本片段的向量 | 3 个向量 |

### 2.2 各存储详解

**① full_docs — 原始文档库**
```python
{
  "doc_id_xxx": {
    "content": "第一章 函数的概念\n1.1 函数的定义...",
    "corpus_id": "math_001",
    "source_file": "高中数学教材"
  }
}
```

**② text_chunks — 分块后的文本**
```python
{
  "chunk_id_yyy": {
    "content": "1.1 函数的定义\n设 A、B 为两个非空数集...",
    "chunk_order_index": 0,
    "chapter": "1.1",           # 章节路径（支持嵌套）
    "domain": ["数学"],          # 所属领域
    "corpus_id": "math_001",
    "source_file": "高中数学教材",
    "ftype": "txt"
  }
}
```

**③ chunk_entity_relation_graph — 实体关系图**
```python
# 用 NetworkX 实现的无向图
# 节点：
Graph.nodes["函数"] = {"entity_type": "concept", "domain": ["数学"], "description": "..."}
Graph.nodes["单调递增"] = {"entity_type": "concept", "domain": ["数学", "函数"], ...}

# 边：
Graph.edges["函数", "单调递增"] = {"type": "Contains", "weight": 1.0, ...}
```

**④ concept_vdb — 实体向量数据库**
```python
# 每个实体的向量表示，用于语义检索
{
  "ent-xxxx": {
    "content": "函数  数学  设 A、B 为两个非空数集...",
    "entity_name": "函数"
  }
  # 向量：文本 → 1536 维向量
}
```

**⑤ relations_vdb — 关系向量数据库**
```python
# 每条关系的向量表示
{
  "rel-xxxx": {
    "content": "Contains 函数 单调递增 单调递增是函数的性质之一",
    "src_id": "函数",
    "tgt_id": "单调递增"
  }
}
```

**⑥ fragment_vdb — 文本片段向量库**
```python
# 每个文本块的向量
{
  "frag-xxxx": {
    "content": "1.1 函数的定义...",
    "chunk_id": "chunk_id_yyy"
  }
}
```

---

## 三、完整数据流：从文档到知识图谱

### 3.1 Insert 流程（插入文档）

以一篇数学文档为例：

```
输入文档（数学教材关于函数的一章）
    ↓
┌─ Step 1: 文档分块（chunking_by_chapter）┐
│   按章节结构切分成 3 个 chunk             │
│   chunk_1: "1.1 函数的定义..."           │
│   chunk_2: "1.2 单调性与奇偶性..."        │
│   chunk_3: "1.3 函数的应用..."           │
└────────────────────────────────────────────┘
    ↓
┌─ Step 2: LLM 实体抽取（extract_entities）─┐
│   对每个 chunk 调用 LLM，提取：           │
│   concepts: 实体列表                       │
│   relations: 关系列表                      │
│                                           │
│   chunk_1 抽取出：                         │
│   - 概念：函数、单调递增、奇偶性           │
│   - 关系：函数→单调递增（Contains）        │
│           函数→奇偶性（Contains）          │
│           单调递增→函数（Depends）         │
└────────────────────────────────────────────┘
    ↓
┌─ Step 3: 图写入（write_nx_graph）────────┐
│   将抽取的实体和关系写入 NetworkX 图       │
│   → 6 个节点，11 条边                      │
└────────────────────────────────────────────┘
    ↓
┌─ Step 4: 向量写入（upsert）──────────────┐
│   concept_vdb:     3 个实体向量            │
│   relations_vdb:  11 个关系向量           │
│   fragment_vdb:    3 个文本块向量          │
└────────────────────────────────────────────┘
```

**实体抽取的 Prompt 示例**（LLM 输入）：
```
-Goal-
Given a text chunk, extract unique concepts (entities) and their relationships...

Text Chunk:
1.1 函数的定义
设 A、B 为两个非空数集，如果对于 A 中的每一个元素 x...

Response Format:
{
  "concepts": [{"name": "...", "domain": [...], "description": "..."}],
  "relations": [{"type": "...", "src": "...", "tgt": "...", "description": "..."}]
}
```

**LLM 返回的 JSON 示例**：
```json
{
  "concepts": [
    {"name": "函数", "domain": ["数学", "函数"], "description": "设 A、B 为两个非空数集..."},
    {"name": "单调递增", "domain": ["数学", "函数"], "description": "如果对于区间 I 上任意两点..."}
  ],
  "relations": [
    {"type": "Contains", "src": "函数", "tgt": "单调递增", "description": "...", "weight": 1.0}
  ]
}
```

### 3.2 Query 流程（查询问答）

HyperGraphRAG 支持三种查询模式：

#### 模式 1: Local（局部查询）
```
问题：函数的基本性质有哪些？

步骤：
1. 从 concept_vdb 检索最相关的实体 → 找到"函数"
2. 从图中找"函数"相邻的实体 → 单调递增、奇偶性
3. 获取相邻文本块 → chunk_2（包含性质详述）
4. 组装上下文 → LLM 生成回答

输出：
{
  "concepts": [{"name": "函数", ...}],
  "relations": [...],
  "text_units": [chunk_2 的内容],
  "response": "函数的基本性质包括：\n1. 单调性...\n2. 奇偶性..."
}
```

#### 模式 2: Global（全局查询）
```
问题：函数的基本性质有哪些？

步骤：
1. 从 relations_vdb 检索最相关的关系 → 找到所有"函数"相关的关系
2. 聚合所有相关关系 → 构建全局上下文
3. 组装上下文 → LLM 生成回答

特点：全局理解，不依赖单个实体，适合需要综合总结的问题
```

#### 模式 3: Hybrid（混合查询）
```
综合 Local 和 Global 的结果，给出最全面的回答
```

---

## 四、Phase 1-4 改造详解

### 4.1 Phase 1: Schema 扩展

**目标**：让 HyperGraphRAG 支持多领域多文档管理

**新增字段**（`TextChunkSchema`）：
```python
"corpus_id": str    # 语料库 ID，同一领域的文档共享
"source_file": str  # 来源文件
"ftype": str        # 文件格式：pdf/txt/md
"chapter": str      # 章节路径，支持"1.2.3"这样的嵌套结构
"domain": list[str] # 所属领域，允许多个
```

**新增查询参数**（`QueryParam`）：
```python
"domains": Optional[list[str]]  # 按领域过滤查询
```

### 4.2 Phase 2: Prompt 升级

**新增 JSON 输出模式**的实体抽取 Prompt（`extract_concepts_and_relations`）：

原来的 Prompt 输出是自由文本，需要解析。新的 Prompt 要求 LLM 直接输出 JSON：
```json
{
  "concepts": [...],
  "relations": [...]
}
```

好处：稳定可靠，不会因为 LLM 输出的格式变化而解析失败。

### 4.3 Phase 3: operate.py 重写

核心逻辑重写，包括：

**extract_entities** — 实体抽取：
- 调用 JSON 模式的抽取 Prompt
- 支持多轮 glean（继续抽取，直到 LLM 说"没有了"）
- 在循环内解析每个 glean 结果并合并，而不是拼接字符串后统一解析

**kg_query** — 知识图谱查询：
- 支持 local / global / hybrid 三种模式
- 按 `domains` 过滤实体和关系

**chunking_by_chapter** — 按章节分块：
- 支持嵌套章节路径（点分格式如 "1.2.3"）
- 优先按章节边界分块，不足时按 token 限制分块

### 4.4 Phase 4: 适配与修复

本阶段的修复最多，列举关键问题：

#### 问题 1: GraphML 不支持 list 类型属性
NetworkX 的 GraphML writer 不支持 `list[str]` 类型。修复：将 `domain: list[str]` 在写入前转为 `"|".join(domain)` 字符串，读取时再转回 list。

#### 问题 2: nano-vectordb 的 query() 只返回 id 和距离
`relations_vdb.query()` 只返回 `{id, distance}`，不返回存储的 `src_id`、`tgt_id`。修复：新增 `get_by_ids()` 方法，直接读取 `client_storage["data"]` 获取完整数据。

#### 问题 3: glean 结果拼接导致 JSON 解析失败
原来的代码 `final_result += glean_result` 把多个 JSON 拼接后再解析。修复：改为在循环内逐个解析并合并到列表。

#### 问题 4: `upsert_edge` 参数数量错误
调用时传了 4 个参数（多了一个 `edge_key`），但签名只接受 3 个。修复：移除多余的参数。

#### 问题 5: `_get_edge_data` 字段映射错误
relations_vdb 存的是 `src_id`/`tgt_id`，但代码访问 `r["hyperedge_name"]`。修复：使用 `get_edge(src_id, tgt_id)` 替代 `get_node(hyperedge_name)`。

#### 问题 6: `meta_fields` 配置错误
relations_vdb 创建时 `meta_fields={"hyperedge_name"}`，但实际存储的字段是 `src_id`/`tgt_id`。修复：改为 `meta_fields={"src_id", "tgt_id", "content"}`。

---

## 五、代码文件对照

| 文件路径 | 作用 | 关键函数/类 |
|----------|------|-------------|
| `hypergraphrag/base.py` | 定义数据模型 | `TextChunkSchema`、`QueryParam`、`BaseVectorStorage.get_by_ids()` |
| `hypergraphrag/storage.py` | 存储实现 | `NanoVectorDBStorage`（向量存储）、`NetworkXStorage`（图存储）、`write_nx_graph()` |
| `hypergraphrag/operate.py` | 核心操作逻辑 | `extract_entities()`、`kg_query()`、`chunking_by_chapter()`、`_get_edge_data()` |
| `hypergraphrag/prompt.py` | Prompt 模板 | `extract_concepts_and_relations` JSON 模式 |
| `hypergraphrag/hypergraphrag.py` | 主入口 | `HyperGraphRAG.insert()`、`HyperGraphRAG.query()` |

---

## 六、测试验证

### 6.1 测试脚本
`test_with_mock_llm.py` — 使用 Mock LLM + Mock Embedding 验证完整数据流，无需真实 LLM API。

### 6.2 验证结果

```
=== insert ===
3 unique concepts after dedup
Writing graph with 6 nodes, 11 edges
→ insert done

=== query ===
mode=local:  ✓（使用 3 个实体、13 个关系、3 个文本块）
mode=global: ✓（使用 0 个实体、11 个关系、3 个文本块）
mode=hybrid: ✓（使用 3 个实体、13 个关系、3 个文本块）
→ all done
```

---

## 七、关键设计决策备忘

1. **chapter 点分嵌套**：`"1.2.3"` 表示第 1 章第 2 节第 3 小节
2. **C 和 KF 的 domain=list[str]**：允许多领域标签
3. **章节优先分块**：先按章节边界切分，不足时按 token 限制再切分
4. **exact-name 去重**：实体去重基于精确名称匹配
5. **relations_vdb 存 src_id/tgt_id**：而不是 hyperedge_name，通过 `get_edge()` 获取完整边数据
6. **Python 3.9 兼容**：使用 `Optional[X]` 而非 `X | None` 语法
