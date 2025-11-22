import os
import re
import time
import json
from typing import Optional, Tuple, List
from tqdm import tqdm
import chromadb
from datasets import load_dataset
from openai import OpenAI
import hashlib

# =========================
# 1. 配置与常量
# =========================

# 本地 Llama Factory API 配置
API_BASE = "http://localhost:8001/v1"
API_KEY = "EMPTY"  # 本地通常不需要 key
MODEL_NAME = "llama-3-8b-instruct" # 根据你启动 API 时指定的 model name 填写

# ChromaDB 路径 (本地持久化)
TRIAN_NUM = 200
TEST_NUM = 50
DB_PATH = f"./causal_memory_db_{TRIAN_NUM}_{TEST_NUM}"

# --- Prompt 模板 ---

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

class CausalAgent:
    def __init__(self):
        # 初始化 LLM 客户端
        self.client = OpenAI(api_key=API_KEY, base_url=API_BASE)
        
        # 初始化向量数据库
        self.chroma_client = chromadb.PersistentClient(path=DB_PATH)
        self.collection = self.chroma_client.get_or_create_collection(name="reasoning_rules")
        
    def generate_stable_id(self, text: str) -> str:
        """
        <--- 新增方法：生成确定性的 ID (MD5)
        只要输入文本一样，跨进程/跨机器生成的 ID 都一样
        """
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def is_cached(self, doc_id: str) -> bool:
        """
        <--- 新增方法：检查 ID 是否已存在于数据库
        """
        existing = self.collection.get(ids=[doc_id])
        # 如果 ids 列表不为空，说明库里有
        return len(existing['ids']) > 0

    def call_llm(self, prompt: str, stop_tokens: List[str] = None, temperature: float = 0.7) -> str:
        """封装 LLM 调用"""
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                stop=stop_tokens,
                max_tokens=512
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM Error: {e}")
            return ""

    def extract_answer(self, text: str) -> Optional[float]:
        """
        从文本中提取最后一个数值作为答案。
        针对 GSM8K 优化：先找 <answer> 标签，如果没有，找最后一个数字。
        """
        # 优先匹配 <answer>Tag</answer>
        tag_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if tag_match:
            text_to_search = tag_match.group(1)
        else:
            text_to_search = text

        # 提取所有数字 (支持整数、小数、负数)
        numbers = re.findall(r"-?\d+\.?\d*", text_to_search.replace(",", ""))
        if not numbers:
            return None
        try:
            return float(numbers[-1])
        except:
            return None
        
    def _safe_extract(self, text: str, tag: str) -> str:
        """
        辅助函数：安全提取 XML 标签内容
        如果找不到标签，或者提取出错，默认返回整个文本或空字符串
        """
        if not text:
            return ""
        
        # 尝试匹配 <tag>content</tag>
        # 注意：增加了 re.IGNORECASE 以防模型输出 <Thought>
        pattern = f"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        
        if match:
            try:
                return match.group(1).strip()
            except IndexError:
                return text # 匹配到了但没拿到 group，返回原文本
        
        # 备选方案：模型可能忘记写闭合标签 </tag>
        # 尝试匹配 <tag> 之后的所有内容
        pattern_open = f"<{tag}>(.*)"
        match_open = re.search(pattern_open, text, re.DOTALL | re.IGNORECASE)
        if match_open:
            return match_open.group(1).strip()
            
        # 如果完全没有标签，直接返回原文本作为 Thought，防止丢失信息
        return text
    
    def causal_filter(self, question: str, gold_ans: float) -> Tuple[bool, str, str]:
        """
        核心算法：因果效应估计
        """
        # Step 1: Generate (X -> M -> Y)
        cot_response = self.call_llm(PROMPT_COT.format(question=question))
        
        # --- 修复点：使用更安全的提取逻辑 ---
        thought = self._safe_extract(cot_response, "thought")
        pred_ans = self.extract_answer(cot_response)

        # Debug 打印：如果你想看模型到底输出了什么导致报错，取消下面注释
        # print(f"DEBUG: Response: {cot_response[:50]}... | Extracted Thought: {thought[:50]}...")

        # Quality Control: 如果连 CoT 都做不对，直接丢弃
        if pred_ans is None:
             return False, thought, "CoT_Format_Error_No_Number"
             
        if abs(pred_ans - gold_ans) > 1e-5:
            return False, thought, "CoT_Incorrect"

        # Step 2: Intervention / Knockout Test (X -> Y_direct)
        # 强制切断 M 的路径
        direct_response = self.call_llm(PROMPT_DIRECT.format(question=question), temperature=0.1)
        direct_ans = self.extract_answer(direct_response)

        # Step 3: Calculate Causal Effect
        # 情况 A: Direct 也对了 -> 说明题目简单/有 Shortcut -> ITE ≈ 0 -> 不存
        if direct_ans is not None and abs(direct_ans - gold_ans) < 1e-5:
            return False, thought, "Spurious_Correlation(Too_Easy)"
        
        # 情况 B: Direct 错了 -> 说明没有 M 做不对 -> ITE > 0 -> 存！
        return True, thought, "Causal_Necessity_Confirmed"

    # def abstract_and_save(self, question: str, valid_thought: str):
    #     """
    #     抽象化并存储 (M -> R)
    #     """
    #     # 1. 抽象化
    #     rule = self.call_llm(PROMPT_ABSTRACT.format(question=question, thought=valid_thought))
        
    #     # 2. 存入向量库
    #     # ID 使用问题的 Hash
    #     doc_id = str(hash(question))
        
    #     self.collection.upsert(
    #         documents=[rule], # 存的是 Rule，之后检索 Rule
    #         metadatas=[{"original_question": question, "source": "gsm8k_train"}],
    #         ids=[doc_id]
    #     )
    #     # 注意：默认 ChromaDB 会自动对 documents 进行 embedding 用于检索
    #     # 如果你想用 Question 做 Query，这里应该把 Question 设为 embedding content，但存 Rule
    #     # 为简化代码，这里假设我们检索时也是用 "Query" 去匹配 "Rule" 的语义 (通常是可行的)
    #     # *更严谨的做法*：self.collection.upsert(embeddings=embed(question), documents=[rule]...)
        
    def abstract_and_save(self, question: str, valid_thought: str, doc_id: str):
        """
        抽象化并存储，使用传入的 stable doc_id
        """
        rule = self.call_llm(PROMPT_ABSTRACT.format(question=question, thought=valid_thought))
        
        self.collection.upsert(
            documents=[rule], 
            metadatas=[{"original_question": question, "source": "gsm8k_train"}],
            ids=[doc_id] # 使用 MD5 ID
        )


    def retrieve_rule(self, query: str, k: int = 1):
        """检索最相关的推理逻辑"""
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        if results['documents'] and len(results['documents'][0]) > 0:
            return results['documents'][0][0] # 返回最相似的一条规则
        return None

def evaluate_standard_cot(agent, test_data):
    """
    运行标准 Zero-shot CoT 基线测试
    不使用任何检索，直接问模型。
    """
    print("\n" + "="*20 + " Baseline: Standard Zero-shot CoT " + "="*20)
    correct = 0
    total = 0
    
    # 用于记录错误样本，方便后续分析
    error_log = []

    for i, item in tqdm(enumerate(test_data), total=len(test_data)):
        question = item['question']
        
        # 提取标准答案数值
        gold_str = item['answer'].split('####')[-1].strip()
        try:
            gold_ans = float(gold_str.replace(',', ''))
        except:
            continue # 跳过格式错误数据

        # --- 核心差异：这里只用标准 CoT Prompt，不查库 ---
        prompt = PROMPT_COT.format(question=question)
        response = agent.call_llm(prompt)
        pred = agent.extract_answer(response)

        # 判定正确性
        is_correct = False
        if pred is not None and abs(pred - gold_ans) < 1e-5:
            correct += 1
            is_correct = True
        
        total += 1
        
        # 如果错了，记录下来，稍后看看你的因果记忆能不能把这些题救回来
        if not is_correct:
            error_log.append({"question": question, "gold": gold_ans, "pred": pred})

    acc = correct / total * 100
    print(f"\n[Baseline Result] Accuracy: {correct}/{total} = {acc:.2f}%")
    return acc, error_log


def main():
    agent = CausalAgent()
    
    # 加载数据
    print("Loading GSM8K dataset...")
    dataset = load_dataset("openai/gsm8k", "main")
    
    # --- 配置实验规模 ---
    # 建议：训练集用 200-500 条挖掘记忆，测试集用 50-100 条验证效果
    train_subset = dataset['train'].select(range(TRIAN_NUM)) 
    test_subset = dataset['test'].select(range(TEST_NUM))    

    # ==========================================
    # Step 1: 运行标准基线 (Standard CoT)
    # ==========================================
    baseline_acc, baseline_errors = evaluate_standard_cot(agent, test_subset)

    # ==========================================
    # Step 2: 因果记忆挖掘 (Training Phase)
    # ==========================================
    print("\n" + "="*20 + " Phase 1: Causal Memory Mining " + "="*20)
    saved_count = 0
    skipped_count = 0
    
    # # 使用 tqdm 显示进度
    # for item in tqdm(train_subset, desc="Mining"):
    #     question = item['question']
    #     gold_str = item['answer'].split('####')[-1].strip()
    #     try:
    #         gold_ans = float(gold_str.replace(',', ''))
    #     except: continue

    #     # 执行因果筛选
    #     is_causal, thought, reason = agent.causal_filter(question, gold_ans)
        
    #     if is_causal:
    #         agent.abstract_and_save(question, thought)
    #         saved_count += 1
    
    # print(f"\n[Mining Done] Saved {saved_count} causal rules from {len(train_subset)} samples.")
    
    for item in tqdm(train_subset, desc="Mining"):
        question = item['question']
        
        # --- 修改点 1: 生成确定性 ID ---
        doc_id = agent.generate_stable_id(question)
        
        # --- 修改点 2: 检查本地库是否已有该记忆 ---
        if agent.is_cached(doc_id):
            # 如果库里有，说明上次跑过且判定为 Valid，直接跳过挖掘
            # (注意：如果上次判定为无效没存，这里会重新跑，但这也是合理的，也许上次跑错了)
            saved_count += 1
            skipped_count += 1
            continue

        # 如果没有缓存，继续跑正常流程
        gold_str = item['answer'].split('####')[-1].strip()
        try:
            gold_ans = float(gold_str.replace(',', ''))
        except: continue

        is_causal, thought, reason = agent.causal_filter(question, gold_ans)
        
        if is_causal:
            agent.abstract_and_save(question, thought, doc_id) # 传入 ID
            saved_count += 1
            
    print(f"\n[Mining Done] Saved {saved_count} causal rules from {len(train_subset)} samples.")

    # ==========================================
    # Step 3: 因果增强测试 (Ours: Causal-CoT)
    # ==========================================
    print("\n" + "="*20 + " Phase 2: Inference with Causal Memory " + "="*20)
    
    correct = 0
    total = 0
    improved_cases = 0 # 记录本来错了，但加了记忆后对了的 Case

    for i, item in enumerate(test_subset):
        question = item['question']
        gold_str = item['answer'].split('####')[-1].strip()
        try:
            gold_ans = float(gold_str.replace(',', ''))
        except: continue

        # 1. 检索
        rule = agent.retrieve_rule(question)
        
        # 2. 构造 Prompt
        if rule:
            # 你的方法：有参考逻辑
            prompt = PROMPT_RAG.format(rule=rule, question=question)
            log_prefix = "[RAG]"
            print(f"\n[RAG Triggered] 使用参考逻辑解题...")
        else:
            # 回退到标准 CoT
            prompt = PROMPT_COT.format(question=question)
            log_prefix = "[Zero]"
            print(f"\n[Zero-shot] 无相关记忆，使用标准 CoT...")

        # 3. 推理
        response = agent.call_llm(prompt)
        pred = agent.extract_answer(response)

        # 4. 统计
        is_correct = (pred is not None and abs(pred - gold_ans) < 1e-5)
        if is_correct:
            correct += 1
        total += 1

        # 5. 关键分析：检查是否修复了基线中的错误
        # 检查这个问题是否在基线的错误列表中
        was_wrong_in_baseline = any(err['question'] == question for err in baseline_errors)
        
        if was_wrong_in_baseline and is_correct:
            improved_cases += 1
            print(f"{log_prefix} SUCCESS! Fixed a hard problem: {question[:30]}...")
        elif was_wrong_in_baseline and not is_correct:
            # 依然做错
            pass 

    # ==========================================
    # Final Report
    # ==========================================
    ours_acc = correct / total * 100
    
    print("\n" + "="*30)
    print(f"FINAL REPORT")
    print("="*30)
    print(f"Baseline Accuracy: {baseline_acc:.2f}%")
    print(f"Ours (Causal) Acc: {ours_acc:.2f}%")
    print(f"Improvement:       {ours_acc - baseline_acc:+.2f}%")
    print(f"Fixed Hard Cases:  {improved_cases} (Problems baseline failed but we solved)")
    print("="*30)

if __name__ == "__main__":
    main()