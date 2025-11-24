import os
import re
import time
import hashlib
import chromadb
from tqdm import tqdm
from datasets import load_dataset
from openai import OpenAI
from typing import Optional, List, Dict

# =========================
# 1. 全局配置
# =========================

API_BASE = "http://localhost:8001/v1"
API_KEY = "EMPTY"
MODEL_NAME = "llama-3-8b-instruct"


# 训练与测试规模配置
TRAIN_SIZE = 1000  # 挖掘用的训练集数量
TEST_SIZE = 1319    # 测试集数量
DB_PATH = f"./unified_experiment_db_{TRAIN_SIZE}_{TEST_SIZE}"  # 统一数据库路径

# --- 新增超参数 ---
RETRIEVAL_K = 3   # 检索到的相关推理过程的个数

# --- Prompts ---
PROMPT_COT = """You are a math reasoning expert.
Question: {question}
Please think step by step to solve this problem. 
Wrap your reasoning process in <thought>...</thought> tags.
Wrap your final numerical answer in <answer>...</answer> tags.
"""

PROMPT_DIRECT = """You are a calculator.
Question: {question}
Give me the final numerical answer directly. 
DO NOT output any reasoning steps, words, or explanations. 
Only output the number.
"""

PROMPT_ABSTRACT = """Analyze the following math problem and its reasoning steps.
Problem: {question}
Reasoning: {thought}

Your task: Extract the **general logical pattern** or **methodology** used to solve this. 
- Remove specific numbers and entities.
- Output ONLY the abstract rule.
"""

# 修改后的 RAG Prompt，支持多条规则输入
PROMPT_RAG = """Reference Logics:
{rules}

Question: {question}
Using the Reference Logics above as guides (you may combine them or select the most relevant one), think step by step to solve the Question.
Wrap your final numerical answer in <answer>...</answer> tags.
"""

# =========================
# 2. 统一智能体类
# =========================

class UnifiedAgent:
    def __init__(self):
        self.client = OpenAI(api_key=API_KEY, base_url=API_BASE)
        self.chroma = chromadb.PersistentClient(path=DB_PATH)
        
        self.naive_col = self.chroma.get_or_create_collection(name="naive_memory")
        self.causal_col = self.chroma.get_or_create_collection(name="causal_memory")

    # --- 工具函数 ---
    def generate_stable_id(self, text: str) -> str:
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def is_processed(self, doc_id: str) -> bool:
        existing = self.naive_col.get(ids=[doc_id])
        return len(existing['ids']) > 0

    def call_llm(self, prompt: str, stop=None, temp=0.0) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp, # 严格控制温度为 0
                stop=stop,
                max_tokens=512
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM Error: {e}")
            return ""

    def extract_answer(self, text: str) -> Optional[float]:
        tag_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        content = tag_match.group(1) if tag_match else text
        numbers = re.findall(r"-?\d+\.?\d*", content.replace(",", ""))
        return float(numbers[-1]) if numbers else None

    def _safe_extract(self, text, tag):
        # 之前修复过的安全提取函数
        match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
        if match: return match.group(1).strip()
        return text # Fallback

    # --- 辅助函数：格式化检索到的规则 ---
    def _format_retrieved_rules(self, documents: List[str]) -> str:
        """
        将检索到的规则列表格式化为带序号的字符串
        """
        if not documents:
            return ""
        formatted = []
        for i, doc in enumerate(documents):
            formatted.append(f"{i+1}. {doc}")
        return "\n".join(formatted)

    # --- 核心：单次遍历，双重挖掘 (逻辑不变) ---
    def run_unified_mining(self, dataset):
        print(f"\n[Phase 1] Starting Unified Mining (Size: {len(dataset)})...")
        
        naive_count = 0
        causal_count = 0
        skipped_count = 0
        
        for item in tqdm(dataset, desc="Mining"):
            question = item['question']
            doc_id = self.generate_stable_id(question)
            
            naive_key = f"naive_{doc_id}"
            causal_key = f"causal_{doc_id}"
            
            if self.is_processed(naive_key):
                skipped_count += 1
                continue

            gold_str = item['answer'].split('####')[-1].strip()
            try: gold_ans = float(gold_str.replace(',', ''))
            except: continue

            resp = self.call_llm(PROMPT_COT.format(question=question))
            pred = self.extract_answer(resp)
            thought = self._safe_extract(resp, "thought")

            if pred is None or abs(pred - gold_ans) > 1e-5:
                continue

            # === 分支 A: 存入 Naive 库 ===
            rule = self.call_llm(PROMPT_ABSTRACT.format(question=question, thought=thought))
            
            self.naive_col.upsert(
                documents=[rule],
                metadatas=[{"question": question, "type": "naive"}],
                ids=[naive_key]
            )
            naive_count += 1

            # === 分支 B: 因果筛选 ===
            direct_resp = self.call_llm(PROMPT_DIRECT.format(question=question))
            direct_pred = self.extract_answer(direct_resp)

            is_necessary = (direct_pred is None or abs(direct_pred - gold_ans) > 1e-5)
            
            if is_necessary:
                self.causal_col.upsert(
                    documents=[rule],
                    metadatas=[{"question": question, "type": "causal"}],
                    ids=[causal_key]
                )
                causal_count += 1
        
        print(f"\nMining Finished.")
        print(f"Skipped (Cached): {skipped_count}")
        print(f"Newly Mined Naive Memories:  {naive_count}")
        print(f"Newly Mined Causal Memories: {causal_count}")

    # --- 核心：三方对比测试 (修改支持 Top-K) ---
    def run_comparative_eval(self, dataset):
        print(f"\n[Phase 2] Starting Comparative Evaluation (Top-{RETRIEVAL_K} Retrieval)...")
        
        results = {
            "Baseline": {"correct": 0, "total": 0},
            "Naive":    {"correct": 0, "total": 0},
            "Causal":   {"correct": 0, "total": 0}
        }
        
        fixed_cases = 0

        for item in tqdm(dataset, desc="Evaluating"):
            question = item['question']
            gold_str = item['answer'].split('####')[-1].strip()
            try: gold_ans = float(gold_str.replace(',', ''))
            except: continue
            
            # --- 1. Baseline (Zero-shot CoT) ---
            resp_base = self.call_llm(PROMPT_COT.format(question=question))
            pred_base = self.extract_answer(resp_base)
            is_base_correct = (pred_base and abs(pred_base - gold_ans) < 1e-5)
            
            results["Baseline"]["total"] += 1
            if is_base_correct: results["Baseline"]["correct"] += 1

            # --- 2. Naive RAG (Top-K) ---
            # 修改点：n_results=RETRIEVAL_K
            naive_res = self.naive_col.query(query_texts=[question], n_results=RETRIEVAL_K)
            retrieved_naive = naive_res['documents'][0] if naive_res['documents'] else []
            
            if retrieved_naive:
                # 修改点：格式化多条规则
                rules_str = self._format_retrieved_rules(retrieved_naive)
                prompt_naive = PROMPT_RAG.format(rules=rules_str, question=question)
            else:
                prompt_naive = PROMPT_COT.format(question=question)
            
            pred_naive = self.extract_answer(self.call_llm(prompt_naive))
            if pred_naive and abs(pred_naive - gold_ans) < 1e-5:
                results["Naive"]["correct"] += 1
            results["Naive"]["total"] += 1

            # --- 3. Causal CoT (Top-K) ---
            # 修改点：n_results=RETRIEVAL_K
            causal_res = self.causal_col.query(query_texts=[question], n_results=RETRIEVAL_K)
            retrieved_causal = causal_res['documents'][0] if causal_res['documents'] else []
            
            if retrieved_causal:
                # 修改点：格式化多条规则
                rules_str = self._format_retrieved_rules(retrieved_causal)
                prompt_causal = PROMPT_RAG.format(rules=rules_str, question=question)
            else:
                prompt_causal = PROMPT_COT.format(question=question)
            
            pred_causal = self.extract_answer(self.call_llm(prompt_causal))
            is_causal_correct = (pred_causal and abs(pred_causal - gold_ans) < 1e-5)
            
            if is_causal_correct: results["Causal"]["correct"] += 1
            results["Causal"]["total"] += 1

            if not is_base_correct and is_causal_correct:
                fixed_cases += 1

        return results, fixed_cases

# =========================
# 3. 主流程
# =========================

def main():
    agent = UnifiedAgent()
    
    print("Loading Dataset...")
    ds = load_dataset("openai/gsm8k", "main")
    train_set = ds['train'].select(range(TRAIN_SIZE))
    test_set = ds['test'].select(range(TEST_SIZE))

    # 1. 统一挖掘
    agent.run_unified_mining(train_set)

    # 2. 统一测试 (Top-K)
    scores, fixed = agent.run_comparative_eval(test_set)

    # 3. 报告
    total = scores["Baseline"]["total"]
    if total == 0: return

    acc_base = scores["Baseline"]["correct"] / total * 100
    acc_naive = scores["Naive"]["correct"] / total * 100
    acc_causal = scores["Causal"]["correct"] / total * 100
    
    storage_naive = agent.naive_col.count()
    storage_causal = agent.causal_col.count()
    storage_ratio = (storage_causal / storage_naive * 100) if storage_naive > 0 else 0

    print("\n" + "#"*50)
    print(f"FINAL REPORT (Retrieval K={RETRIEVAL_K})")
    print("#"*50)
    print(f"Test Set Size: {total}")
    print("-" * 50)
    print(f"{'Method':<15} | {'Accuracy':<10} | {'Storage':<15}")
    print("-" * 50)
    print(f"{'Baseline':<15} | {acc_base:.2f}%     | N/A")
    print(f"{'Naive RAG':<15} | {acc_naive:.2f}%     | {storage_naive:<15}")
    print(f"{'Causal CoT':<15} | {acc_causal:.2f}%     | {storage_causal:<15} ({storage_ratio:.1f}%)")
    print("-" * 50)
    print(f"Improvement over Baseline:  {acc_causal - acc_base:+.2f}%")
    print(f"Improvement over Naive:     {acc_causal - acc_naive:+.2f}%")
    print(f"Hard Cases Fixed:           {fixed}")
    print("#"*50)

if __name__ == "__main__":
    main()
