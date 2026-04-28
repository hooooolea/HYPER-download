# HyperGraphRAG 深度学习总结

## 项目概述

**HyperGraphRAG** 是一个基于超图（Hypergraph）结构知识表示的检索增强生成（RAG）系统，发表在 **NeurIPS 2025**。论文链接：[arXiv:2503.21322](https://arxiv.org/abs/2503.21322)。

核心仓库：[https://github.com/HKUDS/HyperGraphRAG](https://github.com/HKUDS/HyperGraphRAG)

---

## 1. 核心创新：为什么需要超图？

传统知识图谱（Knowledge Graph）中，**一条边只能连接两个实体**（如 `A -> B`）。

HyperGraphRAG 引入**超边（Hyperedge）**的概念——**一条超边可以同时连接多个实体**。这更符合人类知识的实际结构：一个知识片段往往涉及多个实体之间的复杂关系。

**示例**：

```
传统图:  Alex—(observed)—>device  Taylor—(said)—>device

超图:    hyperedge "Taylor observed device" → 包含 Taylor, device 两个实体节点
         超边作为"关系的一等公民"被单独索引，可以被向量检索
```

---

## 2. 整体架构

```
文档输入
  │
  ▼
┌──────────────────────────────────────────────────────┐
│              Insert 流程（建立知识库）                 │
│  1. chunking_by_token_size: 长文本切成 chunk          │
│  2. extract_entities: LLM 提取实体+超边+关系           │
│  3. 三类存储写入: KV存储 / 向量存储 / 图存储          │
└──────────────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────────────┐
│              Query 流程（回答问题）                    │
│  1. kg_query: LLM 从查询中提取 HL(高层) 和 LL(低层) 关键词│
│  2. _build_query_context:                            │
│     local 模式 → 用 LL 关键词查实体向量库             │
│     global 模式 → 用 HL 关键词查超边向量库            │
│     hybrid 模式 → 合并 local + global                │
│  3. 用检索到的实体/超边/原文 chunk 构建上下文          │
│  4. LLM 生成最终答案                                 │
└──────────────────────────────────────────────────────┘
```

---

## 3. 数据模型

### 3.1 三种节点类型

| 类型 | role | 关键字段 | 说明 |
|------|------|---------|------|
| **实体节点** | `entity` | `entity_name`, `entity_type`, `description`, `source_id` | 从文本中提取的命名实体 |
| **超边节点** | `hyperedge` | `hyperedge_name`, `weight`, `source_id` | 知识片段，对应一个完整的语义陈述 |
| **文本块** | - | `content`, `chunk_order_index`, `full_doc_id` | 原始文档的切片 |

### 3.2 关系

```
超边节点 →(边)→ 实体节点

# 存储在 NetworkX 图中
# 超边的 source_id 指向引用的 chunk，用于回溯原文
```

### 3.3 示例数据格式

从 `example_contexts.json` 可见，输入是 ESC 高血压指南的文本段落（医学领域）。

---

## 4. 核心类详解

### 4.1 HyperGraphRAG 主类

**文件**: `hypergraphrag/hypergraphrag.py`

```python
from hypergraphrag import HyperGraphRAG

rag = HyperGraphRAG(
    working_dir="expr/example",        # 工作目录，存放所有存储文件
    kv_storage="JsonKVStorage",        # KV 存储后端（默认 JSON 文件）
    vector_storage="NanoVectorDBStorage",  # 向量数据库后端
    graph_storage="NetworkXStorage",   # 图存储后端（默认 NetworkX）

    # 分块参数
    chunk_token_size=1200,            # 每个 chunk 的 token 数
    chunk_overlap_token_size=100,     # 相邻 chunk 重叠 token 数

    # 实体提取参数
    entity_extract_max_gleaning=2,    # LLM 重复提取轮次（防止遗漏）
    entity_summary_to_max_tokens=500, # 实体描述摘要的最大长度

    # Embedding 参数
    embedding_func=openai_embedding,  # 默认用 OpenAI embedding

    # LLM 参数
    llm_model_func=gpt_4o_mini_complete,
)
```

### 4.2 插入文档

```python
import os, json
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"

rag = HyperGraphRAG(working_dir="expr/example")

with open("example_contexts.json") as f:
    contexts = json.load(f)  # 数组，每项是字符串

rag.insert(contexts)  # 同步接口，内部自动做异步
```

**插入流程内部步骤**：

1. 对每段文本按 token 分块（重叠 100 tokens）
2. 所有 chunk 存入 `text_chunks` KV 存储 + `chunks_vdb` 向量库
3. 调用 `extract_entities()`——**核心 LLM 提取流程**
4. 实体/超边/关系写入 `chunk_entity_relation_graph`（NetworkX 图）
5. 实体描述写入 `entities_vdb` 向量库
6. 超边描述写入 `hyperedges_vdb` 向量库

### 4.3 查询

```python
result = rag.query(
    "老年体弱患者的收缩压 120-129 mmHg 目标证据强度如何？",
    param=QueryParam(
        mode="hybrid",              # local / global / hybrid / naive
        top_k=60,                   # 检索的实体/超边数量
        max_token_for_text_unit=4000,
        max_token_for_global_context=4000,
        max_token_for_local_context=4000,
    )
)
print(result)
```

---

## 5. 实体提取的 LLM 提示词设计

**文件**: `hypergraphrag/prompt.py`

LLM 被要求从每个 chunk 中提取：

### 5.1 超关系（Hyper-relation）

```python
("hyper-relation"<|>"一段描述完整知识的句子"<|>"完整度评分 0-10")
```

### 5.2 实体（Entity）

```python
("entity"<|>"实体名"<|>"实体类型"<|>"实体描述"<|>"重要度评分 0-100")
```

**实体类型默认**：`["organization", "person", "geo", "event", "category"]`

**格式分隔符**：

- `tuple_delimiter`: `<|>`
- `record_delimiter`: `##`
- `completion_delimiter`: `<|COMPLETE|>`

### 5.3 多次提取（Gleaning）

每个 chunk 会重复调用 LLM 最多 `entity_extract_max_gleaning=2` 次，并在每次后询问 `"是否还有遗漏？YES|NO"`，防止 LLM 一次性提取不完整。

---

## 6. 三种查询模式详解

### 6.1 Local 模式（低层）

```
查询 → LLM 提取 LL 关键词（如"血压"、"老年人"）
     → 向量检索 entities_vdb（top_k=60）
     → 获取实体节点 → 找 1-hop 邻居实体
     → 收集关联的原文 chunks
     → 构建上下文 → LLM 生成
```

适合：针对具体实体的问题（如"某药物的副作用是什么"）

### 6.2 Global 模式（高层）

```
查询 → LLM 提取 HL 关键词（如"心血管预后"、"药物疗效"）
     → 向量检索 hyperedges_vdb
     → 获取超边节点 → 找关联的实体
     → 收集关联的原文 chunks
     → 构建上下文 → LLM 生成
```

适合：需要综合多段知识的问题（如"该指南的核心建议是什么"）

### 6.3 Hybrid 模式（混合）

同时执行 local + global，合并去重后一起构建上下文。**这是默认模式**。

---

## 7. 存储后端

### 7.1 三种存储接口

| 存储类型 | 接口基类 | 支持后端 |
|---------|---------|---------|
| KV 存储 | `BaseKVStorage` | JsonKVStorage（默认）、OracleKVStorage、MongoKVStorage、TiDBKVStorage |
| 向量存储 | `BaseVectorStorage` | NanoVectorDBStorage（默认）、ChromaVectorDBStorage、MilvusVectorDBStorge、OracleVectorDBStorage、TiDBVectorDBStorage |
| 图存储 | `BaseGraphStorage` | NetworkXStorage（默认）、Neo4JStorage、OracleGraphStorage |

### 7.2 懒加载外部存储

```python
# 通过 lazy_external_import 实现按需加载
Neo4JStorage = lazy_external_import(".kg.neo4j_impl", "Neo4JStorage")
```

### 7.3 工作目录文件

```
working_dir/
├── kv_store_full_docs.json              # 原始文档
├── kv_store_text_chunks.json            # 文本分块
├── kv_store_llm_response_cache.json     # LLM 响应缓存
├── vdb_entities.json                    # 实体向量库
├── vdb_hyperedges.json                  # 超边向量库
├── vdb_chunks.json                     # chunk 向量库
└── graph_chunk_entity_relation.graphml  # NetworkX 图
```

---

## 8. 评估流程

**文件**: `evaluation/README.md`

评估需要下载额外数据集（从 Terabox 获取），放在 `evaluation/contexts/` 和 `evaluation/datasets/` 目录。

**评估步骤**：

```
Step 1: 构建知识超图
        nohup python script_insert.py --cls hypertension > result_hypertension_insert.log 2>&1 &

Step 2: 检索知识
        python script_hypergraphrag.py --data_source hypertension

Step 3: 基于检索结果生成答案
        python get_generation.py --data_sources hypertension --methods HyperGraphRAG

Step 4: 评估生成质量
        CUDA_VISIBLE_DEVICES=0 python get_score.py --data_source hypertension --method HyperGraphRAG

Step 5: 查看得分
        python see_score.py --data_source hypertension --method HyperGraphRAG
```

对比方法包括：StandardRAG（传统向量检索）、NaiveGeneration（无检索直接生成）。

---

## 9. 关键配置参数一览

| 参数 | 默认值 | 含义 |
|------|-------|------|
| `chunk_token_size` | 1200 | 每个文本块的 token 数 |
| `chunk_overlap_token_size` | 100 | 相邻块重叠 token 数 |
| `entity_extract_max_gleaning` | 2 | LLM 重复提取轮数 |
| `entity_summary_to_max_tokens` | 500 | 实体描述摘要最大长度 |
| `llm_model_max_async` | 16 | LLM 最大并发数 |
| `embedding_func_max_async` | 16 | Embedding 最大并发数 |
| `embedding_batch_num` | 32 | Embedding 批大小 |
| `node_embedding_algorithm` | "node2vec" | 节点 embedding 算法（预留） |

---

## 10. 使用建议与注意事项

1. **API Key 配置**：必须设置 `os.environ["OPENAI_API_KEY"]`，或使用支持 OpenAI 接口的代理服务

2. **工作目录**：每个 `HyperGraphRAG` 实例会创建独立目录存放存储文件，重复使用相同 `working_dir` 会自动加载已有数据

3. **批量插入**：文档以数组形式传入，内部自动分块、并发 LLM 提取

4. **模式选择**：
   - 简单实体问题 → `local`
   - 需要综合全局知识 → `global`
   - 通用场景 → `hybrid`（默认）

5. **缓存**：默认开启 LLM 响应缓存（`enable_llm_cache=True`），相同查询直接返回缓存结果

6. **自定义知识注入**：支持 `insert_custom_kg()` 方法直接插入结构化知识（实体、关系、超边），绕过 LLM 提取

---

## 11. 文件结构一览

```
HyperGraphRAG/
├── README.md                          # 项目说明
├── requirements.txt                   # 依赖列表
├── example_contexts.json             # 示例数据（ESC 高血压指南）
├── script_construct.py                # 快速构建脚本
├── script_query.py                   # 快速查询脚本
├── hypergraphrag/
│   ├── __init__.py                   # 导出 HyperGraphRAG, QueryParam
│   ├── base.py                       # 所有存储基类定义
│   ├── hypergraphrag.py              # 主类 HyperGraphRAG
│   ├── llm.py                        # LLM 调用封装（OpenAI / HuggingFace）
│   ├── operate.py                    # 核心操作：分块、实体提取、查询
│   ├── prompt.py                     # 所有提示词模板
│   ├── storage.py                    # 默认存储实现（JsonKV、NanoVectorDB、NetworkX）
│   ├── utils.py                      # 工具函数（tokenization、hash、logger）
│   └── kg/                           # 多种数据库后端实现
│       ├── neo4j_impl.py
│       ├── milvus_impl.py
│       ├── mongo_impl.py
│       ├── oracle_impl.py
│       ├── chroma_impl.py
│       └── tidb_impl.py
└── evaluation/                       # 评估流程
    ├── README.md                     # 评估步骤说明
    ├── contexts/                     # 评估用上下文数据
    ├── datasets/                     # 评估用问答数据
    ├── script_insert.py
    ├── script_hypergraphrag.py
    ├── get_generation.py
    ├── get_score.py
    └── see_score.py
```
