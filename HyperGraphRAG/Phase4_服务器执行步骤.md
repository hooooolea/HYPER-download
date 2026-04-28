# Phase 4 服务器执行步骤

> 执行前请确保已在服务器上 `cd ~/Desktop/auto-aiwork/hermes_doc/HyperGraphRAG`

---

## 第一步：语法检查

```bash
cd ~/Desktop/auto-aiwork/hermes_doc/HyperGraphRAG

python3 -m py_compile \
  hypergraphrag/hypergraphrag.py \
  hypergraphrag/operate.py \
  hypergraphrag/base.py \
  hypergraphrag/utils.py \
  hypergraphrag/prompt.py

echo "=== syntax check exit: $? ==="
```

预期：`=== syntax check exit: 0 ===`

---

## 第二步：模块导入检查（AST 解析）

```bash
python3 -c "
import sys, ast

files = [
    'hypergraphrag/hypergraphrag.py',
    'hypergraphrag/operate.py',
    'hypergraphrag/base.py',
    'hypergraphrag/utils.py',
    'hypergraphrag/prompt.py',
]

ok = True
for f in files:
    with open(f) as fh:
        try:
            ast.parse(fh.read())
            print(f'OK  {f}')
        except SyntaxError as e:
            print(f'ERR {f}: {e}')
            ok = False

sys.exit(0 if ok else 1)
"
echo "=== import check exit: $? ==="
```

预期：5 个文件全部 `OK`，最终 `exit: 0`

---

## 第三步：创建测试文件

在服务器上创建 `test_phase4.py`（内容如下），放在 `~/Desktop/auto-aiwork/hermes_doc/HyperGraphRAG/test_phase4.py`

```python
# test_phase4.py
import sys
sys.path.insert(0, '.')

from hypergraphrag import HyperGraphRAG
from hypergraphrag.base import QueryParam

# 初始化（用临时目录，禁用 LLM/embedding 以验证数据流）
rag = HyperGraphRAG(
    working_dir="/tmp/hypergraphrag_test_phase4",
    llm_model_func=None,
    embedding_func=None,
)

# 准备带完整元信息的文档
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
try:
    rag.insert(docs)
    print("=== insert done ===")
except Exception as e:
    print(f"=== insert ERROR: {e} ===")
    import traceback; traceback.print_exc()

print("=== query start ===")
for mode in ["local", "global", "hybrid"]:
    try:
        result = rag.query(f"函数是什么？ mode={mode}", QueryParam(mode=mode))
        print(f"mode={mode}: {str(result)[:80] if result else 'None'}")
    except Exception as e:
        print(f"mode={mode} ERROR: {e}")
print("=== all query modes done ===")
```

---

## 第四步：执行测试

```bash
cd ~/Desktop/auto-aiwork/hermes_doc/HyperGraphRAG
python3 test_phase4.py
echo "=== test exit: $? ==="
```

预期：
- `=== insert done ===` 出现（无异常中断）
- 3 种 query 模式均有输出（内容可能为空字符串，因 LLM 为 None）
- `=== test exit: 0 ===`

---

## 第五步：汇报结果

将每一步的完整输出发给助手（我），由我判断是否通过。

### 通过标准

| 步骤 | 通过条件 |
|------|---------|
| 第一步 | exit code = 0，无任何输出 |
| 第二步 | 5 个文件全部 `OK`，exit code = 0 |
| 第四步 | insert 完成，3 种 query 均有输出，exit code = 0 |

### 如果有报错

把完整输出粘贴过来，标注是哪一步。
