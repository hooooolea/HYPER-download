"""
Phase 4 验证脚本：使用 Mock LLM + Mock Embedding 验证完整数据流
"""
import sys
sys.path.insert(0, '.')

import json as _json
import uuid
from hypergraphrag import HyperGraphRAG
from hypergraphrag.base import QueryParam
import numpy as np

# ============================================================
# Mock LLM
# ============================================================
async def mock_llm(prompt, history_messages=None, **kwargs):
    """根据 prompt 首词判断调用类型"""
    content = str(prompt) if prompt else ""
    first_word = content.split()[0] if content else ""

    # glean continuation prompt: "MANY knowdge fragements..." → no
    if first_word == "MANY":
        print(f"[MOCK] MANY prompt → 'no'")
        return "no"

    # glean if_loop prompt: "Please check..." → no
    if first_word == "Please":
        print(f"[MOCK] Please prompt → 'no'")
        return "no"

    # extract_concepts_and_relations prompt: "-Goal-\nGiven a text chunk..."
    if first_word == "-Goal-":
        data = {
            "concepts": [
                {"name": "函数", "domain": ["数学", "函数"],
                 "description": "设 A、B 为两个非空数集，如果对于 A 中的每一个元素 x，按某种确定的法则 f，在 B 中都有唯一确定的元素 y 与之对应，则称 f 为定义在 A 上的函数。",
                 "source_chunks": []},
                {"name": "单调递增", "domain": ["数学", "函数"],
                 "description": "如果对于区间 I 上任意两点 x1 < x2，有 f(x1) < f(x2)，则称 f 在 I 上单调递增。",
                 "source_chunks": []},
                {"name": "奇偶性", "domain": ["数学", "函数"],
                 "description": "如果对于定义域内任意 x，有 f(-x) = f(x)，则称 f 为偶函数；若 f(-x) = -f(x)，则称 f 为奇函数。",
                 "source_chunks": []}
            ],
            "relations": [
                {"type": "Contains", "src": "函数", "tgt": "单调递增",
                 "description": "单调递增是函数的性质之一", "weight": 1.0},
                {"type": "Contains", "src": "函数", "tgt": "奇偶性",
                 "description": "奇偶性是函数的性质之一", "weight": 1.0},
                {"type": "Depends", "src": "单调递增", "tgt": "函数",
                 "description": "单调递增的概念依赖于函数定义", "weight": 0.8},
                {"type": "Related", "src": "单调递增", "tgt": "奇偶性",
                 "description": "单调性和奇偶性都是函数的性质", "weight": 0.6}
            ]
        }
        result = _json.dumps(data, ensure_ascii=False)
        print(f"[MOCK] -Goal- extract → {len(result)} chars")
        return result

    # high_level_keywords
    if "high_level_keyword" in content or "hl_keyword" in content:
        return _json.dumps({"high_level_keywords": ["函数", "单调性", "奇偶性"]}, ensure_ascii=False)

    # rag_response
    if "context_data" in content or "rag_response" in content:
        return "Mock LLM: 根据检索到的上下文，函数是数学中的基本概念。"

    # summarize
    if "summarize" in content.lower():
        return "Mock LLM: 这是一段关于函数基本概念的数学文本。"

    # 默认
    print(f"[MOCK] DEFAULT (first_word={repr(first_word)}) → fallback")
    return _json.dumps({
        "concepts": [{"name": "函数", "domain": ["数学"], "description": "数学基本概念"}],
        "relations": []
    }, ensure_ascii=False)

# ============================================================
# Mock Embedding
# ============================================================
class MockEmbedding:
    embedding_dim: int = 1536

    async def __call__(self, texts):
        return np.random.rand(len(texts), self.embedding_dim).astype(np.float32)

# ============================================================
# 测试
# ============================================================
print("=== 初始化 ===")
rag = HyperGraphRAG(
    working_dir=f"/tmp/hypergraphrag_mock_test_{uuid.uuid4().hex[:8]}",
    llm_model_func=mock_llm,
    embedding_func=MockEmbedding(),
)

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

print("\n=== insert ===")
try:
    rag.insert(docs)
    print("=== insert done ===")
except Exception as e:
    print(f"=== insert ERROR: {e} ===")
    import traceback; traceback.print_exc()

print("\n=== query ===")
for mode in ["local", "global", "hybrid"]:
    try:
        result = rag.query(f"函数的基本性质有哪些？", QueryParam(mode=mode))
        print(f"mode={mode}: {str(result)[:80] if result else 'None'}")
    except Exception as e:
        print(f"mode={mode} ERROR: {e}")
        import traceback; traceback.print_exc()

print("\n=== all done ===")
