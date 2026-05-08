import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse
import asyncio

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hypergraphrag.llm import openai_complete_if_cache


def generate_response(d):
    prompt = f"""You are a helpful assistant. You are given the retrieved knowledge and a question. Your task is to generate a comprehensive and accurate answer based on the retrieved knowledge. You should synthesize information from multiple sources when available.

---Retrieved Knowledge---
{d['knowledge']}

---Question---
{d['question']}

When you have the final answer, you can output the answer inside <answer>...</answer>.

Output format for answer:
<think>
...
</think>

<answer>
...
</answer>
"""
    d['prompt'] = prompt
    try:
        response = asyncio.run(
            openai_complete_if_cache(
                model="llama3.1:8b",
                prompt=prompt,
                system_prompt=None,
                base_url="http://localhost:11434/v1",
                api_key="ollama",
                temperature=0.7,
                caching=True,
                cache=None,
            )
        )
        d['generation'] = response.strip()
    except Exception as e:
        d['generation'] = f"[ERROR] {str(e)}"
    return d


def process_method(method, data_sources):
    for data_source in data_sources:
        print(f"[DEBUG] {method} {data_source}")
        data_dir = f"results/{method}/{data_source}/test_knowledge.json"
        with open(data_dir) as f:
            data = json.load(f)

        results = []
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(generate_response, d) for d in data]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"{method}"):
                results.append(future.result())

        save_dir = f"results/{method}/{data_source}/test_generation.json"
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        with open(save_dir, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"[{method}] Results saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_sources', type=str, default='hypertension')
    parser.add_argument('--methods', type=str, default='HyperGraphRAG')
    args = parser.parse_args()

    data_sources = [s.strip() for s in args.data_sources.split(',')]
    methods = [m.strip() for m in args.methods.split(',')]

    for method in methods:
        print(f"[DEBUG] Processing method: {method}")
        process_method(method, data_sources)


if __name__ == "__main__":
    main()
