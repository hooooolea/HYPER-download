#!/usr/bin/env python
"""测试 MiniMax LLM 是否正常工作"""
import sys, os, asyncio
sys.path.insert(0, '.')

from hypergraphrag.llm import minimax_complete_if_cache

async def test_llm():
    print("=== 测试 MiniMax LLM ===")
    try:
        result = await minimax_complete_if_cache(
            prompt="你好，请简单介绍一下函数的概念。",
            model="MiniMax-M2.7",
        )
        print(f"LLM 返回: {result[:200]}")
        return True
    except Exception as e:
        print(f"LLM 调用失败: {e}")
        return False

asyncio.run(test_llm())
