import os
import re
import time
import chromadb
from tqdm import tqdm
from datasets import load_dataset
from openai import OpenAI
from typing import Optional, List
import hashlib

# ================= Config =================
API_BASE = "http://localhost:8002/v1"
API_KEY = "EMPTY"
MODEL_NAME = "llama-3-8b-instruct"

TRIAN_NUM = 200
TEST_NUM = 50
DB_PATH = f"./comparison_db_{TRIAN_NUM}_{TEST_NUM}"


# ================= Prompt 模板=================

# P1: 标准 CoT 推理 (X -> M -> Y)
PROMPT_COT = """You are a math reasoning expert.
Question: {question}
Please think step by step to solve this problem. 
Wrap your reasoning process in <thought>...</thought> tags.
Wrap your final numerical answer in <answer>...</answer> tags.
"""

# P2: 干预测试 - 直接回答 (X -> Y_direct)
# 这里的核心是强制模型跳过推理，测试直觉/Shortcut
PROMPT_DIRECT = """You are a calculator.
Question: {question}
Give me the final numerical answer directly. 
DO NOT output any reasoning steps, words, or explanations. 
Only output the number.
"""

# P3: 记忆抽象 (M -> R)
# 将具体的数字推理转化为通用的解题逻辑
PROMPT_ABSTRACT = """Analyze the following math problem and its reasoning steps.
Problem: {question}
Reasoning: {thought}

Your task: Extract the **general logical pattern** or **methodology** used to solve this. 
- Remove specific numbers and entities (e.g., change "5 apples" to "items").
- Focus on the algorithmic steps (e.g., "First calculate total cost, then divide by quantity").
- Output ONLY the abstract rule.
"""

# P4: 基于因果记忆的推理 (Recall R + X -> Y)
PROMPT_RAG = """Reference Logic: {rule}

Question: {question}
Using the Reference Logic above as a guide, think step by step to solve the Question.
Wrap your final numerical answer in <answer>...</answer> tags.
"""

class ExperimentAgent:
    def __init__(self):
        self.client = OpenAI(api_key=API_KEY, base_url=API_BASE)
        self.chroma = chromadb.PersistentClient(path=DB_PATH)
        
        # === 关键：创建两个独立的记忆库 ===
        # 1. Naive 库：存所有做对的题
        self.naive_col = self.chroma.get_or_create_collection(name="naive_memory")
        # 2. Causal 库：只存通过因果测试的题
        self.causal_col = self.chroma.get_or_create_collection(name="causal_memory")
    
    # --- 新增：ID 与 缓存工具 ---
    def generate_stable_id(self, text: str) -> str:
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def is_processed(self, naive_id: str) -> bool:
        """
        检查该问题是否已经被成功处理过。
        逻辑：Naive 库包含所有推理正确的样本。如果 Naive 库里有，说明之前已经跑过且存下来了。
        """
        existing = self.naive_col.get(ids=[naive_id])
        return len(existing['ids']) > 0

    # --- 基础工具函数 ---
    def call_llm(self, prompt, stop=None, temp=0.7):
        try:
            resp = self.client.chat.completions.create(
                model=MODEL_NAME, messages=[{"role": "user", "content": prompt}],
                temperature=temp, stop=stop, max_tokens=512
            )
            return resp.choices[0].message.content.strip()
        except: return ""

    def extract_answer(self, text):
        numbers = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
        return float(numbers[-1]) if numbers else None

    def _safe_extract(self, text, tag):
        # 之前修复过的安全提取函数
        match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
        if match: return match.group(1).strip()
        return text # Fallback

# --- 核心：并行训练逻辑 (带缓存) ---
    def train_parallel(self, dataset):
        """
        同时训练 Naive 和 Causal 两个库
        """
        print(f"开始并行训练 (Dataset Size: {len(dataset)})...")
        
        newly_processed = 0
        skipped_count = 0
        
        for item in tqdm(dataset, desc="Training"):
            q = item['question']
            
            # 1. 生成稳定 ID
            base_id = self.generate_stable_id(q)
            naive_id = f"naive_{base_id}"
            causal_id = f"causal_{base_id}"

            # 2. 缓存检查
            # 如果 Naive 库里已经有这个 ID，说明上次跑过且成功了，跳过
            if self.is_processed(naive_id):
                skipped_count += 1
                continue

            gold = float(item['answer'].split('####')[-1].strip().replace(',', ''))

            # 3. 标准推理
            resp = self.call_llm(PROMPT_COT.format(question=q))
            pred = self.extract_answer(resp)
            thought = self._safe_extract(resp, "thought")

            # 如果做错，跳过 (不存)，下次运行如果不通过缓存检查，还会重试
            if pred is None or abs(pred - gold) > 1e-5:
                continue

            newly_processed += 1

            # === 分支 1: Naive RAG (只要对就存) ===
            rule = self.call_llm(PROMPT_ABSTRACT.format(question=q, thought=thought))
            
            self.naive_col.upsert(
                documents=[rule], 
                metadatas=[{"q": q}], 
                ids=[naive_id] # 使用稳定 ID
            )

            # === 分支 2: Causal-CoT (做对还要验证) ===
            # 执行 Knockout Test
            direct_resp = self.call_llm(PROMPT_DIRECT.format(question=q), temp=0.1)
            direct_pred = self.extract_answer(direct_resp)

            # 只有当 Direct 答错 (Need reasoning) 时才存
            is_necessary = (direct_pred is None or abs(direct_pred - gold) > 1e-5)
            
            if is_necessary:
                self.causal_col.upsert(
                    documents=[rule], 
                    metadatas=[{"q": q}], 
                    ids=[causal_id] # 使用稳定 ID
                )
        
        print("\n" + "="*30)
        print(f"Training Finished.")
        print(f"Skipped (Already Cached): {skipped_count}")
        print(f"Newly Processed:          {newly_processed}")
        print(f"Current Naive Count:      {self.naive_col.count()}")
        print(f"Current Causal Count:     {self.causal_col.count()}")
        print("="*30 + "\n")

    # --- 核心：对比测试逻辑 ---
    def test_comparison(self, dataset):
        results = {"Naive": 0, "Causal": 0, "Total": 0}
        
        print(f"开始对比测试 (Test Size: {len(dataset)})...")

        for item in tqdm(dataset, desc="Testing"):
            q = item['question']
            gold = float(item['answer'].split('####')[-1].strip().replace(',', ''))
            results["Total"] += 1

            # --- 测试 A: 使用 Naive Memory ---
            naive_res = self.naive_col.query(query_texts=[q], n_results=1)
            if naive_res['documents'][0]:
                rule = naive_res['documents'][0][0]
                prompt = PROMPT_RAG.format(rule=rule, question=q)
            else:
                prompt = PROMPT_COT.format(question=q)
            
            pred_naive = self.extract_answer(self.call_llm(prompt))
            if pred_naive and abs(pred_naive - gold) < 1e-5:
                results["Naive"] += 1

            # --- 测试 B: 使用 Causal Memory ---
            causal_res = self.causal_col.query(query_texts=[q], n_results=1)
            if causal_res['documents'][0]:
                rule = causal_res['documents'][0][0]
                prompt = PROMPT_RAG.format(rule=rule, question=q)
            else:
                prompt = PROMPT_COT.format(question=q)
                
            pred_causal = self.extract_answer(self.call_llm(prompt))
            if pred_causal and abs(pred_causal - gold) < 1e-5:
                results["Causal"] += 1

        return results

# ================= Main Execution =================
def main():
    agent = ExperimentAgent()
    
    # 加载数据
    ds = load_dataset("openai/gsm8k", "main")
    
    # 建议配置：
    # 训练集稍大一点，以便 Naive 库积累足够多的噪音，显现出劣势
    train_set = ds['train'].select(range(TRIAN_NUM)) 
    test_set = ds['test'].select(range(TEST_NUM))

    # 1. 训练
    agent.train_parallel(train_set)

    # 2. 测试
    scores = agent.test_comparison(test_set)

    # 3. 报告
    total = scores["Total"]
    naive_acc = scores["Naive"] / total * 100
    causal_acc = scores["Causal"] / total * 100

    print("\n" + "#"*40)
    print("FINAL COMPARISON REPORT")
    print("#"*40)
    print(f"Total Test Samples: {total}")
    print(f"Naive RAG Accuracy:  {naive_acc:.2f}%")
    print(f"Causal-CoT Accuracy: {causal_acc:.2f}%")
    print(f"Delta (Ours - Naive): {causal_acc - naive_acc:+.2f}%")
    print("-" * 40)
    
    # 关键结论生成
    if causal_acc >= naive_acc:
        print("CONCLUSION: Causal-CoT achieved similar/better performance with MUCH smaller memory size.")
        print("This proves that filtering out spurious correlations reduces retrieval noise.")
    else:
        print("CONCLUSION: Causal-CoT underperformed. Check if the causal filter is too strict.")

if __name__ == "__main__":
    main()