# Author 版本修改计划

> 目标：将 author 版本（原仓库 ~/Desktop/HyperGraphRAG-author/）的 LLM 改为 Ollama llama3.1:8b，embedding 改为智谱 zhipu_embedding。参考仓库：~/Desktop/auto-aiwork/hermes_doc/HyperGraphRAG/

---

## 背景说明

- author 版本在 `hypergraphrag/llm.py` 中已有 `ollama_model_complete`（584行）和 `zhipu_embedding`（735行），无需新增函数
- 只需改 import、默认值、和调用方式
- 改动后 push 到 GitHub，服务器 pull 并跑评测

---

## 改动清单

### 1. `hypergraphrag/hypergraphrag.py`

| 位置 | 原内容 | 改后内容 |
|------|--------|----------|
| import | `from .llm import openai_embedding` | `from .llm import zhipu_embedding` |
| import | `from .llm import gpt_4o_mini_complete` | `from .llm import ollama_model_complete` |
| HyperGraphRAG dataclass | `embedding_func: typing.Any = field(default_factory=lambda: openai_embedding)` | `embedding_func: typing.Any = field(default_factory=lambda: zhipu_embedding)` |
| HyperGraphRAG dataclass | `llm_model_func: typing.Any = field(default_factory=lambda: gpt_4o_mini_complete)` | `llm_model_func: typing.Any = field(default_factory=lambda: ollama_model_complete)` |
| HyperGraphRAG dataclass | `llm_model_name: str = "meta-llama/Llama-3.2-1B-Instruct"` | `llm_model_name: str = "llama3.1:8b"` |

---

### 2. `evaluation/script_insert.py`

- **删除** `os.environ["OPENAI_API_KEY"] = open("openai_api_key.txt").read().strip()`
- **删除** `client = OpenAI(...)` 整段
- （import 中的 `from openai import OpenAI` 也删除）

---

### 3. `evaluation/script_hypergraphrag.py`

- **删除** `os.environ["OPENAI_API_KEY"] = open("openai_api_key.txt").read().strip()`
- **删除** `client = OpenAI(...)` 整段
- （import 中的 `from openai import OpenAI` 也删除）

---

### 4. `evaluation/eval_g.py`

| 位置 | 改动 |
|------|------|
| import | 删除 `from openai import OpenAI` |
| import | 删除 `os.environ["OPENAI_API_KEY"] = ...` |
| import | 删除 `client = OpenAI(...)` |
| import | 新增 `from hypergraphrag.llm import openai_complete_if_cache` |
| import | 新增 `import asyncio` |
| 调用处 | `client.chat.completions.create(model="gpt-4o-mini", messages=..., temperature=0)` → `asyncio.run(openai_complete_if_cache(model="llama3.1:8b", base_url="http://localhost:11434/v1", api_key="ollama", prompt=prompt, temperature=0, system_prompt=system_prompt))` |
| max_workers | 保持 `7` 不变 |

---

### 5. `evaluation/get_generation.py`

| 位置 | 改动 |
|------|------|
| import | 删除 `from openai import OpenAI` |
| import | 删除 `os.environ["OPENAI_API_KEY"] = ...` |
| import | 删除 `client = OpenAI(...)` |
| import | 新增 `from hypergraphrag.llm import openai_complete_if_cache` |
| import | 新增 `import asyncio` |
| 调用处 | `client.chat.completions.create(model="gpt-4o-mini", ...)` → `asyncio.run(openai_complete_if_cache(...))` |
| max_workers | `32` → `8` |

---

### 6. `evaluation/get_score.py`

| 位置 | 改动 |
|------|------|
| 文件顶部 | 新增 `import nest_asyncio; nest_asyncio.apply()` |
| 文件顶部 | 新增 `to_python_float()` 函数（用于 JSON float 序列化） |
| max_workers | `64` → `16` |

---

## 验证步骤

1. 本地确认 `hypergraphrag/llm.py` 中 `zhipu_embedding` 和 `ollama_model_complete` 存在
2. 确认服务器 Ollama 已运行 `ollama run llama3.1:8b`
3. push 到 GitHub 后，服务器 pull
4. 服务器执行 `ulimit -n 65535` 后跑评测
5. 对比 author 版本和我们的版本的 score 曲线

---

## 改动后目录结构

```
HyperGraphRAG-author/
├── hypergraphrag/
│   ├── __init__.py
│   ├── llm.py              # 不改（已有 ollama_model_complete + zhipu_embedding）
│   └── hypergraphrag.py    # 改：默认 LLM/embedding 改为 ollama + zhipu
├── evaluation/
│   ├── script_insert.py           # 改：去掉 OpenAI client，用 zhipu_embedding
│   ├── script_hypergraphrag.py    # 改：去掉 OpenAI client
│   ├── eval_g.py                  # 改：gpt-4o-mini → ollama
│   ├── get_generation.py          # 改：gpt-4o-mini → ollama
│   └── get_score.py               # 改：加 nest_asyncio，max_workers 调低
└── MODIFY-PLAN.md                 # 本文档
```
