#!/usr/bin/env python
"""测试 MiniMax LLM + 智谱 Embedding 的 HyperGraphRAG"""
import sys
sys.path.insert(0, '.')

from hypergraphrag import HyperGraphRAG, QueryParam

# 初始化
rag = HyperGraphRAG(
    working_dir="/tmp/hypergraphrag_test_minimax",
    # 所有参数都用默认值，即 MiniMax M2.7 + 智谱 embedding-3
)

# 准备文档
docs = [
    {
        "content": """第一章 函数的概念
1.1 函数的定义
设 A、B 为两个非空数集，如果对于 A 中的每一个元素 x，按某种确定的法则 f，在 B 中都有唯一确定的元素 y 与之对应，则称 f 为定义在 A 上的函数。

1.2 函数的性质
1.2.1 单调性
如果对于区间 I 上任意两点 x1 < x2，有 f(x1) < f(x2)，则称 f 在 I 上单调递增。

1.2.2 奇偶性
如果对于定义域内任意 x，有 f(-x) = f(x)，则称 f 为偶函数；若 f(-x) = -f(x)，则称 f 为奇函数。""",
        "corpus_id": "高中数学-必修一",
        "source_file": "第一章.pdf",
        "ftype": "pdf",
        "domain": ["数学", "函数"],
        "chapter_hint": '[{"chapter":"1","content":"第一章 函数的概念"},{"chapter":"1.1","content":"1.1 函数的定义"},{"chapter":"1.2","content":"1.2 函数的性质"}]',
    }
]

print("=== insert start ===")
rag.insert(docs)
print("=== insert done ===")

print("\n=== query start ===")
result = rag.query("函数是什么？", QueryParam(mode="hybrid"))
print("result:", result[:200] if result else "None")
print("=== query done ===")
