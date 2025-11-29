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


# 规模配置
# CSQA train有9741条，validation有1221条
TRAIN_SIZE = 1000  # 挖掘用的训练集数量
TEST_SIZE = 1221    # 验证集测试数量

# 超参数
RETRIEVAL_K = 3   
NEIGHBOR_CHECK_K = 3
NEIGHBOR_PASS_MIN = 2 

DB_PATH = f"./csqa_{NEIGHBOR_CHECK_K}_neighbor_db_{TRAIN_SIZE}_{TEST_SIZE}"  # CSQA 专用数据库路径

# --- Prompts 适配 CSQA ---
PROMPT_COT = """You are a commonsense reasoning expert.
{input_text}
Please think step by step about the relationship between the concepts in the question and the choices.
Wrap your reasoning process in <thought>...</thought> tags.
Wrap your FINAL choice letter (e.g., A) in <answer>...</answer> tags.
"""

PROMPT_DIRECT = """You are a quiz machine.
{input_text}
Select the correct option letter directly.
DO NOT output any reasoning steps, words, or explanations. 
Only output the choice letter wrapped in <answer>...</answer> tags.
"""

PROMPT_ABSTRACT = """Analyze the following commonsense problem and its reasoning.
{input_text}
Reasoning: {thought}

Your task: Extract the **general commonsense knowledge** or **relationship** used to solve this. 
- Example: "If you want to keep warm, you should use a blanket."
- Do NOT mention specific option letters (like A or B).
- Output ONLY the abstract knowledge.
"""

PROMPT_RAG = """Reference Knowledge:
{rules}

{input_text}
Using the Reference Knowledge above as a guide, think step by step to select the best option.
Wrap your final choice letter in <answer>...</answer> tags.
"""

# =========================
# 2. 统一智能体类
# =========================

class UnifiedCSQAAgent:
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
        """检查 Naive 库是否已有 (包含 correct 和 failed)"""
        existing = self.naive_col.get(ids=[doc_id])
        return len(existing['ids']) > 0

    def call_llm(self, prompt: str, stop=None, temp=0.0) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=MODEL_NAME, messages=[{"role": "user", "content": prompt}],
                temperature=temp, stop=stop, max_tokens=256 # CSQA 不需要太长 token
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM Error: {e}")
            return ""

    def format_csqa_input(self, question, choices):
        """格式化输入：拼接 Question 和 Options"""
        labels = choices['label'] # ['A', 'B', 'C'...]
        texts = choices['text']   # ['cat', 'dog', ...]
        
        formatted = f"Question: {question}\nChoices:\n"
        for l, t in zip(labels, texts):
            formatted += f"({l}) {t}\n"
        return formatted

    def extract_choice(self, text: str) -> Optional[str]:
        """提取答案字母 (A/B/C/D/E)"""
        # 1. 尝试标签
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
        if match:
            content = match.group(1).strip().upper()
            # 提取第一个出现的字母 A-E
            valid = re.search(r"[A-E]", content)
            if valid: return valid.group(0)
            
        # 2. Fallback: 找最后一个单独的大写字母
        # 排除类似 "The answer is A" 的文本干扰，简单正则
        candidates = re.findall(r"\b([A-E])\b", text.upper())
        if candidates:
            return candidates[-1]
            
        return None

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
            q_text = item['question'] # 仅用问题文本做检索索引
            gold_label = item['answerKey']
            
            # 为了验证邻居，我们需要把该题的所有信息存入 metadata
            # Chroma metadata 只能存 string/int/float，不能存 list
            # 所以我们要把 choices 序列化
            choices_labels = ",".join(item['choices']['label'])
            choices_texts = "|".join(item['choices']['text'])
            
            ids.append(self.generate_stable_id(q_text))
            docs.append(q_text)
            metadatas.append({
                "question": q_text,
                "gold": gold_label,
                "c_labels": choices_labels,
                "c_texts": choices_texts
            })
            
            if len(ids) >= 100:
                self.neighbor_pool.upsert(ids=ids, documents=docs, metadatas=metadatas)
                ids, docs, metadatas = [], [], []
        if ids: self.neighbor_pool.upsert(ids=ids, documents=docs, metadatas=metadatas)

    # --- 查找 K 个邻居 ---
    def find_nearest_neighbors(self, question: str, k: int):
        results = self.neighbor_pool.query(query_texts=[question], n_results=k + 2)
        neighbors = []
        if not results['documents']: return neighbors
        
        candidates = results['metadatas'][0]
        
        for cand in candidates:
            if cand['question'].strip() == question.strip(): continue
            
            # 反序列化 choices
            c_labels = cand['c_labels'].split(',')
            c_texts = cand['c_texts'].split('|')
            choices_dict = {'label': c_labels, 'text': c_texts}
            
            neighbors.append({
                "question": cand['question'],
                "choices": choices_dict,
                "answerKey": cand['gold']
            })
            if len(neighbors) >= k: break
        return neighbors

    # --- 核心挖掘流程 ---
    def run_unified_mining(self, dataset):
        self.prepare_neighbor_pool(dataset)
        print(f"\n[Phase 1] Mining CSQA (Checks: {NEIGHBOR_CHECK_K})...")
        
        stats = {"naive_saved": 0, "knockout_fail": 0, "neighbor_fail": 0, "neighbor_pass": 0, "skipped": 0, "failed_attempts": 0}
        
        for item in tqdm(dataset, desc="Mining"):
            # 格式化输入
            question_text = item['question']
            formatted_input = self.format_csqa_input(question_text, item['choices'])
            gold_label = item['answerKey']
            
            doc_id = self.generate_stable_id(question_text)
            naive_key = f"naive_{doc_id}"
            causal_key = f"causal_{doc_id}"
            
            # 1. 缓存检查
            if self.is_processed(naive_key):
                stats["skipped"] += 1
                continue

            # 2. 生成 CoT
            resp = self.call_llm(PROMPT_COT.format(input_text=formatted_input))
            pred = self.extract_choice(resp)
            # thought = self._safe_extract_thought(resp)
            thought = self._safe_extract(resp, "thought")

            # 3. 基础过滤 + 错误缓存 (防止死循环)
            if pred != gold_label:
                # 记录为失败，防止下次重复跑
                self.naive_col.upsert(
                    documents=["FAILED"],
                    metadatas=[{"question": question_text, "type": "failed"}],
                    ids=[naive_key]
                )
                stats["failed_attempts"] += 1
                continue

            # === Save Naive ===
            rule = self.call_llm(PROMPT_ABSTRACT.format(input_text=formatted_input, thought=thought))
            self.naive_col.upsert(
                documents=[rule],
                metadatas=[{"question": question_text, "type": "naive"}],
                ids=[naive_key]
            )
            stats["naive_saved"] += 1

            # === Knockout Test ===
            direct_resp = self.call_llm(PROMPT_DIRECT.format(input_text=formatted_input))
            direct_pred = self.extract_choice(direct_resp)
            
            # 如果不思考也能对，说明是常识捷径 (Too Easy) -> 剔除
            is_necessary = (direct_pred != gold_label)
            
            if not is_necessary:
                stats["knockout_fail"] += 1
                continue 

            # === Neighbor Verification ===
            neighbors = self.find_nearest_neighbors(question_text, k=NEIGHBOR_CHECK_K)
            passed_count = 0
            
            if not neighbors:
                is_robust = True
            else:
                for neigh in neighbors:
                    neigh_input = self.format_csqa_input(neigh['question'], neigh['choices'])
                    # 用 Rule 解邻居
                    prompt_verify = PROMPT_RAG.format(rules=f"1. {rule}", input_text=neigh_input)
                    verify_pred = self.extract_choice(self.call_llm(prompt_verify))
                    
                    if verify_pred == neigh['answerKey']:
                        passed_count += 1
                is_robust = (passed_count >= NEIGHBOR_PASS_MIN)

            if is_robust:
                self.causal_col.upsert(
                    documents=[rule],
                    metadatas=[{"question": question_text, "type": "causal"}],
                    ids=[causal_key]
                )
                stats["neighbor_pass"] += 1
            else:
                stats["neighbor_fail"] += 1
        
        print(f"\n[Mining Report]")
        print(f"Stats: {stats}")

    # --- 对比测试 (Validation Set) ---
    def run_comparative_eval(self, dataset):
        print(f"\n[Phase 2] Evaluation on Validation Set...")
        results = {k: {"correct": 0, "total": 0} for k in ["Baseline", "Naive", "Causal"]}
        
        for item in tqdm(dataset, desc="Eval"):
            formatted_input = self.format_csqa_input(item['question'], item['choices'])
            gold_label = item['answerKey']
            
            # 1. Baseline
            pred_base = self.extract_choice(self.call_llm(PROMPT_COT.format(input_text=formatted_input)))
            if pred_base == gold_label: results["Baseline"]["correct"] += 1
            results["Baseline"]["total"] += 1
            
            # 2. Naive
            naive_res = self.naive_col.query(query_texts=[item['question']], n_results=RETRIEVAL_K)
            docs = naive_res['documents'][0] if naive_res['documents'] else []
            p_naive = PROMPT_RAG.format(rules=self._format_retrieved_rules(docs), input_text=formatted_input) if docs else PROMPT_COT.format(input_text=formatted_input)
            if self.extract_choice(self.call_llm(p_naive)) == gold_label: results["Naive"]["correct"] += 1
            results["Naive"]["total"] += 1
            
            # 3. Causal
            causal_res = self.causal_col.query(query_texts=[item['question']], n_results=RETRIEVAL_K)
            docs = causal_res['documents'][0] if causal_res['documents'] else []
            p_causal = PROMPT_RAG.format(rules=self._format_retrieved_rules(docs), input_text=formatted_input) if docs else PROMPT_COT.format(input_text=formatted_input)
            if self.extract_choice(self.call_llm(p_causal)) == gold_label: results["Causal"]["correct"] += 1
            results["Causal"]["total"] += 1

        return results

def main():
    agent = UnifiedCSQAAgent()
    
    print("Loading CSQA Dataset...")
    ds = load_dataset("tau/commonsense_qa", "default")
    
    # Train 用于挖掘
    train_set = ds['train'].select(range(TRAIN_SIZE))
    # Validation 用于测试
    test_set = ds['validation'].select(range(TEST_SIZE))

    agent.run_unified_mining(train_set)
    scores = agent.run_comparative_eval(test_set)

    total = scores["Baseline"]["total"]
    if total == 0: return

    print("\n" + "#"*50)
    print("CSQA FINAL REPORT")
    print("#"*50)
    for method in ["Baseline", "Naive", "Causal"]:
        acc = scores[method]["correct"] / total * 100
        print(f"{method} Accuracy: {acc:.2f}%")
    
    storage_naive = agent.naive_col.count()
    storage_causal = agent.causal_col.count()
    print("-" * 50)
    print(f"Storage Ratio: {storage_causal}/{storage_naive} ({(storage_causal/storage_naive*100) if storage_naive else 0:.1f}%)")

if __name__ == "__main__":
    main()
