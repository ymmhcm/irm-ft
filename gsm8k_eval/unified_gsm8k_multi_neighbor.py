import os
import re
import time
import hashlib
import chromadb
from tqdm import tqdm
from datasets import load_dataset
from openai import OpenAI
from typing import Optional, List, Dict, Tuple

# =========================
# 1. 全局配置
# =========================

API_BASE = "http://localhost:8001/v1"
API_KEY = "EMPTY"
MODEL_NAME = "llama-3-8b-instruct"

# 训练与测试规模
TRAIN_SIZE = 600
TEST_SIZE = 1319

# --- 检索 RAG 的参数 ---
RETRIEVAL_K = 1  

# --- 邻居验证的参数 (Invariance Settings) ---
NEIGHBOR_CHECK_K = 3  # 验证时找最近的 N 个邻居
NEIGHBOR_PASS_MIN = 2 # N 个邻居中至少要解对 M 个才算通过 (容错机制)

DB_PATH = f"./unified_{NEIGHBOR_CHECK_K}_neighbor_db_{TRAIN_SIZE}_{TEST_SIZE}"  # 数据库路径

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

PROMPT_RAG = """Reference Logics:
{rules}

Question: {question}
Using the Reference Logics above as guides, think step by step to solve the Question.
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
        self.neighbor_pool = self.chroma.get_or_create_collection(name="neighbor_pool")

    # --- 工具函数 ---
    def generate_stable_id(self, text: str) -> str:
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def is_processed(self, doc_id: str) -> bool:
        existing = self.naive_col.get(ids=[doc_id])
        return len(existing['ids']) > 0

    def call_llm(self, prompt: str, stop=None, temp=0.0) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=MODEL_NAME, messages=[{"role": "user", "content": prompt}],
                temperature=temp, stop=stop, max_tokens=512
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

    # def _safe_extract_thought(self, text: str) -> str:
    #     match = re.search(r"(.*)", text, re.DOTALL | re.IGNORECASE)
    #     if match_open: return match_open.group(1).strip()
    #     return text
    def _safe_extract(self, text, tag):
        match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
        if match: return match.group(1).strip()
        return text # Fallback

    def _format_retrieved_rules(self, documents: List[str]) -> str:
        if not documents: return ""
        return "\n".join([f"{i+1}. {doc}" for i, doc in enumerate(documents)])

    # --- 邻居池准备 ---
    def prepare_neighbor_pool(self, dataset):
        if self.neighbor_pool.count() >= len(dataset):
            print("[Info] Neighbor pool already populated.")
            return

        print("[Phase 0] Populating Neighbor Pool...")
        ids, docs, metadatas = [], [], []
        
        for item in tqdm(dataset, desc="Indexing"):
            q = item['question']
            gold = item['answer'].split('####')[-1].strip()
            ids.append(self.generate_stable_id(q))
            docs.append(q)
            metadatas.append({"question": q, "gold_ans": gold})
            
            if len(ids) >= 100:
                self.neighbor_pool.upsert(ids=ids, documents=docs, metadatas=metadatas)
                ids, docs, metadatas = [], [], []
        if ids: self.neighbor_pool.upsert(ids=ids, documents=docs, metadatas=metadatas)

    # --- 升级版：查找 K 个邻居 ---
    def find_nearest_neighbors(self, question: str, k: int) -> List[Tuple[str, float]]:
        """
        查找最相似的 K 个邻居 (排除自身)
        为了防止 Top-1 是自己，我们检索 K+2 个，然后过滤
        """
        results = self.neighbor_pool.query(query_texts=[question], n_results=k + 2)
        
        neighbors = []
        if not results['documents']: return neighbors
        
        candidates = results['metadatas'][0]
        
        for cand in candidates:
            # 简单的自排除逻辑：如果文本完全一样，跳过
            if cand['question'].strip() == question.strip():
                continue
                
            try:
                g = float(cand['gold_ans'].replace(',', ''))
                neighbors.append((cand['question'], g))
            except: continue
            
            # 找够了就停
            if len(neighbors) >= k:
                break
                
        return neighbors

    # --- 核心：双重挖掘 (多邻居验证) ---
    def run_unified_mining(self, dataset):
        self.prepare_neighbor_pool(dataset)
        
        print(f"\n[Phase 1] Mining with Multi-Neighbor Verification (Checks: {NEIGHBOR_CHECK_K}, Pass: {NEIGHBOR_PASS_MIN})...")
        
        stats = {
            "naive_saved": 0,
            "knockout_fail": 0,
            "neighbor_fail": 0,
            "neighbor_pass": 0,
            "skipped_cached": 0
        }
        
        for item in tqdm(dataset, desc="Mining"):
            question = item['question']
            doc_id = self.generate_stable_id(question)
            naive_key = f"naive_{doc_id}"
            causal_key = f"causal_{doc_id}"
            
            if self.is_processed(naive_key):
                stats["skipped_cached"] += 1
                continue

            gold_str = item['answer'].split('####')[-1].strip()
            try: gold_ans = float(gold_str.replace(',', ''))
            except: continue

            # 1. CoT
            resp = self.call_llm(PROMPT_COT.format(question=question))
            pred = self.extract_answer(resp)
            thought = self._safe_extract(resp, "thought")

            is_correct = (pred is not None and abs(pred - gold_ans) < 1e-5)

            if not is_correct:
                # [关键修改] 即使做错了，也要存入数据库占位，防止下次重复跑
                # 我们存入一个空的 rule 或者标记为 failed
                self.naive_col.upsert(
                    documents=["FAILED_ATTEMPT"], # 占位符
                    metadatas=[{"question": question, "type": "failed"}],
                    ids=[naive_key]
                )
                # 统计一下（可选）
                # stats["failed_attempts"] += 1 
                continue

            # 如果做对了，继续下面的流程 (抽象 -> 存入 Naive -> Knockout ...)

            # Save Naive
            rule = self.call_llm(PROMPT_ABSTRACT.format(question=question, thought=thought))
            self.naive_col.upsert(
                documents=[rule],
                metadatas=[{"question": question, "type": "naive"}],
                ids=[naive_key]
            )
            stats["naive_saved"] += 1

            # 2. Knockout Test
            direct_resp = self.call_llm(PROMPT_DIRECT.format(question=question))
            direct_pred = self.extract_answer(direct_resp)
            is_necessary = (direct_pred is None or abs(direct_pred - gold_ans) > 1e-5)
            
            if not is_necessary:
                stats["knockout_fail"] += 1
                continue 

            # 3. Multi-Neighbor Verification
            neighbors = self.find_nearest_neighbors(question, k=NEIGHBOR_CHECK_K)
            
            passed_count = 0
            # 如果没有邻居(数据太少)，默认通过或跳过，这里选择默认通过
            if not neighbors:
                is_robust = True
            else:
                for n_q, n_gold in neighbors:
                    # 验证每一个邻居
                    prompt_verify = PROMPT_RAG.format(rules=f"1. {rule}", question=n_q)
                    verify_resp = self.call_llm(prompt_verify)
                    verify_pred = self.extract_answer(verify_resp)
                    
                    if verify_pred is not None and abs(verify_pred - n_gold) < 1e-5:
                        passed_count += 1
                
                # 检查通过率
                is_robust = (passed_count >= NEIGHBOR_PASS_MIN)

            if is_robust:
                self.causal_col.upsert(
                    documents=[rule],
                    metadatas=[{"question": question, "type": "causal"}],
                    ids=[causal_key]
                )
                stats["neighbor_pass"] += 1
            else:
                stats["neighbor_fail"] += 1
        
        print(f"\n[Mining Report]")
        print(f"Skipped (Cached):    {stats['skipped_cached']}")
        print(f"Naive Memory Saved:  {stats['naive_saved']}")
        print(f"Knockout Rejected:   {stats['knockout_fail']}")
        print(f"Neighbor Rejected:   {stats['neighbor_fail']} (Failed < {NEIGHBOR_PASS_MIN}/{NEIGHBOR_CHECK_K} checks)")
        print(f"Final Causal Saved:  {stats['neighbor_pass']}")

    # --- 对比测试 ---
    def run_comparative_eval(self, dataset):
        print(f"\n[Phase 2] Evaluation (Retrieval K={RETRIEVAL_K})...")
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
            
            # Baseline
            resp_base = self.call_llm(PROMPT_COT.format(question=question))
            pred_base = self.extract_answer(resp_base)
            is_base_correct = (pred_base and abs(pred_base - gold_ans) < 1e-5)
            if is_base_correct: results["Baseline"]["correct"] += 1
            results["Baseline"]["total"] += 1

            # Naive
            naive_res = self.naive_col.query(query_texts=[question], n_results=RETRIEVAL_K)
            docs = naive_res['documents'][0] if naive_res['documents'] else []
            prompt_naive = PROMPT_RAG.format(rules=self._format_retrieved_rules(docs), question=question) if docs else PROMPT_COT.format(question=question)
            pred_naive = self.extract_answer(self.call_llm(prompt_naive))
            if pred_naive and abs(pred_naive - gold_ans) < 1e-5: results["Naive"]["correct"] += 1
            results["Naive"]["total"] += 1

            # Causal
            causal_res = self.causal_col.query(query_texts=[question], n_results=RETRIEVAL_K)
            docs = causal_res['documents'][0] if causal_res['documents'] else []
            prompt_causal = PROMPT_RAG.format(rules=self._format_retrieved_rules(docs), question=question) if docs else PROMPT_COT.format(question=question)
            pred_causal = self.extract_answer(self.call_llm(prompt_causal))
            is_causal_correct = (pred_causal and abs(pred_causal - gold_ans) < 1e-5)
            if is_causal_correct: results["Causal"]["correct"] += 1
            results["Causal"]["total"] += 1

            if not is_base_correct and is_causal_correct: fixed_cases += 1

        return results, fixed_cases

def main():
    agent = UnifiedAgent()
    ds = load_dataset("openai/gsm8k", "main")
    train_set = ds['train'].select(range(TRAIN_SIZE))
    test_set = ds['test'].select(range(TEST_SIZE))

    agent.run_unified_mining(train_set)
    scores, fixed = agent.run_comparative_eval(test_set)

    total = scores["Baseline"]["total"]
    if total == 0: return

    acc_base = scores["Baseline"]["correct"] / total * 100
    acc_naive = scores["Naive"]["correct"] / total * 100
    acc_causal = scores["Causal"]["correct"] / total * 100
    
    storage_naive = agent.naive_col.count()
    storage_causal = agent.causal_col.count()

    print("\n" + "#"*50)
    print(f"FINAL REPORT (Multi-Neighbor {NEIGHBOR_PASS_MIN}/{NEIGHBOR_CHECK_K})")
    print("#"*50)
    print(f"Baseline Accuracy:   {acc_base:.2f}%")
    print(f"Naive RAG Accuracy:  {acc_naive:.2f}%")
    print(f"Causal CoT Accuracy: {acc_causal:.2f}%")
    print("-" * 50)
    print(f"Storage Ratio:       {(storage_causal/storage_naive*100) if storage_naive else 0:.1f}%")
    print(f"Hard Cases Fixed:    {fixed}")
    print("#"*50)

if __name__ == "__main__":
    main()
