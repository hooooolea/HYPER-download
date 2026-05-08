# HyperGraphRAG 评测项目概述

## 项目背景

评测 HyperGraphRAG 在 5 个数据集上的表现，对比基线方法。指标：EM、F1、R-Sim (SimCSE)、Gen (LLM打分)。

数据集：hypertension、agriculture、cs、legal、mix

## 当前进度

| 数据集 | Insert | Query | Generation | Scoring |
|--------|--------|-------|------------|---------|
| hypertension | ✅ | 进行中 | ⏳ | ⏳ |
| agriculture | ⏳ | ⏳ | ⏳ | ⏳ |
| cs | ⏳ | ⏳ | ⏳ | ⏳ |
| legal | ⏳ | ⏳ | ⏳ | ⏳ |
| mix | ⏳ | ⏳ | ⏳ | ⏳ |

## 技术架构

- **LLM**: Ollama + llama3.1:8b（本地推理，OpenAI 兼容 API）
- **Embedding**: ZhipuAI API（智谱 Embedding，有 429 余额问题）
- **评测流程**: script_insert → script_hypergraphrag → get_generation → get_score
- **工作目录**: `~/HYPER-download/evaluation/expr/{dataset}/`

## 评测流程

### Step 1: Insert（构建知识图谱）

```bash
# 服务器上
source ~/HYPER-download/venv/bin/activate
cd ~/HYPER-download/evaluation
python script_insert.py --cls hypertension
```

输出：实体（concepts）、关系（relations）、片段（fragments）的图结构和向量索引

### Step 2: HyperGraphRAG Query

```bash
python script_hypergraphrag.py --data_source hypertension
```

### Step 3: Generation

```bash
python get_generation.py --data_source hypertension
```

### Step 4: Scoring

```bash
python get_score.py --data_source hypertension
```

## 下一步计划

1. hypertension query 跑完后，运行 generation + scoring
2. 其余 4 个 dataset（agriculture、cs、legal、mix）各跑一遍完整的 4 步流程
3. 汇总 5 个 dataset 的 EM、F1、R-Sim、Gen 指标结果

## 关键配置

### script_insert.py

```python
rag = HyperGraphRAG(
    working_dir=WORKING_DIR,
    embedding_func_max_async=32,
    llm_model_max_async=16,
    llm_model_name="llama3.1:8b",
    llm_model_kwargs={
        "base_url": "http://localhost:11434/v1",
        "api_key": "***"
    }
)
```

### script_hypergraphrag.py

```python
rag = HyperGraphRAG(
    working_dir=f"expr/{data_source}",
    llm_model_name="llama3.1:8b",
    llm_model_kwargs={
        "base_url": "http://localhost:11434/v1",
        "api_key": "***"
    }
)
```

## 遇到的问题及解决方案

### 问题1: JSON 格式输出不稳定

**现象**: llama3.1:8b 不遵循 JSON 格式指令，输出截断的 JSON 导致 `json.JSONDecodeError`

**原因**: 指令遵循能力弱，JSON 输出经常不完整

**解决方案**: 
- 改用纯文本格式 `CONCEPTS: ... | RELATIONS: ...`
- 用正则表达式解析输出（见 operate.py fallback 逻辑）

### 问题2: LLM 连接错误 (APIConnectionError)

**现象**: `httpx.ConnectError: All connection attempts failed`

**原因**: 默认连接 OpenAI API，没有指向本地 Ollama

**解决方案**: 
- 设置 `llm_model_name="llama3.1:8b"`
- 设置 `llm_model_kwargs={"base_url": "http://localhost:11434/v1", "api_key": "***"}`

### 问题3: model 参数冲突（tenacity + partial bug）

**现象**: `TypeError: openai_complete_if_cache() got multiple values for argument 'model'`

**原因**: HyperGraphRAG 用 `functools.partial` 包装 `openai_complete_if_cache` 绑定 `model` 参数。tenacity retry 的 `copy()` 处理 partial 时会把第一个位置参数同时用 positional 和 keyword 方式传递，导致冲突。

**解决方案**: 在 `HyperGraphRAG.__post_init__` 里用普通 async wrapper 函数代替 partial 包装 `llm_model_func`。

### 问题4: 实体抽取断点续跑

**现象**: Ctrl+C 中断后重新运行会从头开始抽取实体（225 chunks 需要 40+ 分钟）

**原因**: graphml 文件只在 insert 完全成功后写入

**解决方案**: 在 `hypergraphrag.py` 的 `ainsert` 开头加 checkpoint 判断：graph 已有节点时跳过抽取步骤。

### 问题5: ZhipuAI Embedding API 429 错误

**现象**: embedding 阶段大量 429 retry

**原因**: 智谱 API 余额不足

**解决方案**: 
- 使用 tenacity retry 机制自动重试
- 或者充值智谱 API 账户

### 问题6: 缺少 llm_model_max_token_size 参数

**现象**: `KeyError: 'llm_model_max_token_size'`

**原因**: `_handle_entity_relation_summary` 需要这个参数但 dataclass 未定义

**解决方案**: 在 `hypergraphrag.py` 的 HyperGraphRAG dataclass 中添加：
```python
llm_model_max_token_size: int = 2048
```

### 问题7: llm_kwargs 属性名拼写错误

**现象**: `AttributeError: 'HyperGraphRAG' object has no attribute 'llm_kwargs'`

**原因**: dataclass 字段名是 `llm_model_kwargs`，但 `__post_init__` 里写成了 `self.llm_kwargs`（少了 `_model`）

**解决方案**: 修复 `hypergraphrag.py` 第 244-245 行：
```python
_base_url = self.llm_model_kwargs.get("base_url")
_api_key = self.llm_model_kwargs.get("api_key")
```

### 问题8: contexts 和 datasets 软链接问题

**现象**: 软链接指向 Mac 本地路径 `/Users/ejuer/Desktop/...`，在服务器上无效

**原因**: 早期建软链接时指向了错误路径

**解决方案**: 
- 删除软链接
- 用 scp 直接从 Mac 传数据到服务器

### 问题9: JSON parsing error 日志噪音

**现象**: 日志里频繁出现 `JSON parsing error: Expecting value: line 1 column 1`

**原因**: llama3.1:8b 输出 markdown 代码块（` ```python ... ` 或 ` ```text ... `）而不是纯文本，JSON 解析失败

**状态**: 非 bug，fallback 正则解析正常接管，不影响功能。日志吵杂但不影响正确性，暂不处理。

## 数据文件

- contexts: `~/HYPER-download/evaluation/contexts/{dataset}_contexts.json`
- questions: `~/HYPER-download/evaluation/datasets/{dataset}/`

## Ollama 服务管理

```bash
# 启动
ollama serve

# 或后台运行
nohup ollama serve > ~/ollama.log 2>&1 &

# 检查状态
curl -s http://localhost:11434/api/tags
```

## Git 工作流

所有代码修改在 Mac 本地 `~/Desktop/auto-aiwork/hermes_doc/HyperGraphRAG/` 进行，通过 git push 同步到 GitHub，服务器上 git pull 拉取。

代码修改后我主动 push，不等用户提醒。

```bash
# Mac 本地（我来执行）
git add -A && git commit -m "message" && git push

# 服务器上（用户执行）
cd ~/HYPER-download && git pull
```
