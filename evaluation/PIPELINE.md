# HyperGraphRAG 评测流程

## 环境准备

### 服务器信息
- SSH: `ssh -p 23 root@106.75.68.167`
- 工作目录: `~/HYPER-download/evaluation/`
- Python 环境: `source ~/HYPER-download/venv/bin/activate`

### Ollama（LLM）
```bash
# 启动（前台运行，占终端）
ollama serve

# 或后台运行
nohup ollama serve > ~/ollama.log 2>&1 &

# 验证
curl -s http://localhost:11434/api/tags
```

### 环境变量
```bash
export ZHIPUAI_API_KEY="your_zhipu_api_key"  # embedding 用
```

## 数据文件

从 Mac 传到服务器：
```bash
scp -r -P 23 ~/Desktop/auto-aiwork/hermes_doc/HyperGraphRAG-contexts\&datasets/contexts root@106.75.68.167:/root/HYPER-download/evaluation/
scp -r -P 23 ~/Desktop/auto-aiwork/hermes_doc/HyperGraphRAG-contexts\&datasets/datasets/* root@106.75.68.167:/root/HYPER-download/evaluation/datasets/
```

## 评测流程（4 步）

### Step 1: Insert（构建知识图谱）

```bash
source ~/HYPER-download/venv/bin/activate
cd ~/HYPER-download/evaluation
python script_insert.py --cls hypertension
```

输出: 19 concepts, 21 relations, 225 fragments → graph with 244 nodes, 19 edges

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

## 数据集

- hypertension、agriculture、cs、legal、mix
- contexts: `contexts/{dataset}_contexts.json`
- questions: `datasets/{dataset}/questions.json`

## 常见问题

### 1. TypeError: openai_complete_if_cache() got multiple values for argument 'model'

**原因**: `llm_model_kwargs` 里不要传 `model`，model 只通过 `llm_model_name` 传

**正确配置**:
```python
rag = HyperGraphRAG(
    working_dir=f"expr/{data_source}",
    llm_model_name="llama3.1:8b",
    llm_model_kwargs={
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama"
    }
)
```

### 2. JSON 解析错误

**原因**: llama3.1:8b 不遵循 JSON 格式指令

**解决**: 已改为纯文本格式 + 正则解析

### 3. Embedding 429 错误

**原因**: 智谱 API 余额不足

**解决**: tenacity retry 会自动重试，或充值

### 4. Ctrl+C 后重跑要重新抽取实体

**解决**: 已加 checkpoint 逻辑，graph 已有节点时跳过抽取

## Git 工作流

所有修改在 Mac 本地进行：

```bash
# Mac 本地
cd ~/Desktop/auto-aiwork/hermes_doc/HyperGraphRAG
git add -A && git commit -m "message" && git push

# 服务器上
cd ~/HYPER-download && git pull
```
