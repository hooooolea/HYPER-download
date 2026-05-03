#!/usr/bin/env python
"""诊断 embedding 维度和 VDB 配置"""
import sys, os, asyncio
sys.path.insert(0, '.')

from hypergraphrag.llm import zhipu_embedding
from hypergraphrag import HyperGraphRAG

async def diag():
    print("=== 诊断 zhipu_embedding 维度 ===")
    texts = ["这是一个测试文本"]
    result = await zhipu_embedding(texts)  # 必须 await
    print(f"返回类型: {type(result)}")
    print(f"返回形状: {result.shape}")
    print(f"期望维度: 1024")
    print(f"实际维度: {result.shape[1] if result.ndim == 2 else 'N/A'}")

    print("\n=== 诊断 fragment_vdb embedding_dim 配置 ===")
    rag = HyperGraphRAG(working_dir="/tmp/hypergraphrag_diag")
    print(f"HyperGraphRAG.embedding_dim = {rag.embedding_dim}")
    print(f"fragment_vdb._embedding_dim = {rag.fragment_vdb._embedding_dim}")

    vdb_embedding_dim = rag.fragment_vdb._embedding_dim
    if result.shape[1] != vdb_embedding_dim:
        print(f"\n!!! 维度不匹配 !!!")
        print(f"embedding 实际维度: {result.shape[1]}, VDB 期望维度: {vdb_embedding_dim}")
    else:
        print(f"\n维度匹配: OK")

asyncio.run(diag())
