# HyperGraphRAG Evaluation

> 评测流程文档见 [PIPELINE.md](./PIPELINE.md)

## 快速开始

```bash
# 1. 启动 Ollama
ollama serve

# 2. 设置环境变量
export ZHIPUAI_API_KEY="your_key"

# 3. Insert
source ~/HYPER-download/venv/bin/activate
cd ~/HYPER-download/evaluation
python script_insert.py --cls hypertension

# 4. Query
python script_hypergraphrag.py --data_source hypertension

# 5. Generation
python get_generation.py --data_source hypertension

# 6. Score
python get_score.py --data_source hypertension
```

## 数据文件

从 Mac 传到服务器：
```bash
scp -r -P 23 ~/Desktop/auto-aiwork/hermes_doc/HyperGraphRAG-contexts\&datasets/contexts root@106.75.68.167:/root/HYPER-download/evaluation/
scp -r -P 23 ~/Desktop/auto-aiwork/hermes_doc/HyperGraphRAG-contexts\&datasets/datasets/* root@106.75.68.167:/root/HYPER-download/evaluation/datasets/
```

