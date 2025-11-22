#!/usr/bin/env python3
"""
local_hf_gsm8k_eval.py
- 读取 HF datasets 的 openai/gsm8k test
- 使用本地 HF 模型生成答案（transformers）
- 支持 greedy (deterministic) 和 self-consistency / maj@k（sampling）
- 输出 CSV 结果并打印 accuracy
Requirements:
pip install transformers datasets accelerate torch
Usage examples:
# greedy
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 python local_hf.py --model_name_or_path ../catastrophic_forgetting/Llama3-8B --mode greedy
# maj@k sampling
python local_hf_gsm8k_eval.py --model_name_or_path /path/to/your/model --mode maj --k 16 --temperature 0.8 --max_examples 100
"""
import argparse
import re
import json
from collections import Counter, defaultdict
from fractions import Fraction
from decimal import Decimal, InvalidOperation
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def build_prompt(question, few_shot_text=None):
    """
    Default prompt: zero-shot chain-of-thought style.
    If few_shot_text (string) provided, it will be prepended.
    """
    tail = "\nLet's think step by step.\n\nThe answer is"
    if few_shot_text:
        return few_shot_text.strip() + "\n\nQ: " + question.strip() + "\nA:" + tail
    else:
        return "Q: " + question.strip() + "\nA:" + tail

_NUMBER_RE = re.compile(r'(-?\d+(?:,\d{3})*(?:\.\d+)?(?:e[+-]?\d+)?|-?\d+/\d+)')
_FRACTION_RE = re.compile(r'(-?\d+)\s*/\s*(\d+)')

def canonicalize_numeric(text):
    """
    Try to extract the final numeric answer from model text.
    Returns a canonical string and a numeric object (Fraction or Decimal) when possible.
    Strategy:
      - find all fraction patterns "a/b" and decimals/ints (with commas)
      - pick the *last* matched token (common heuristic)
      - if it's fraction, convert to Fraction
      - else convert to Decimal after removing commas
    """
    if not text:
        return None, None
    matches = _NUMBER_RE.findall(text)
    if not matches:
        return None, None
    last = matches[-1]
    # try fraction first
    m = _FRACTION_RE.match(last)
    if m:
        num = int(m.group(1))
        den = int(m.group(2))
        try:
            frac = Fraction(num, den)
            return str(frac), frac
        except Exception:
            pass
    # else decimal / int (remove commas)
    cleaned = last.replace(',', '')
    try:
        dec = Decimal(cleaned)
        # remove trailing .0 if integer-like for canonical string
        if dec == dec.to_integral():
            return str(int(dec)), dec
        else:
            # normalize decimal (remove exponent if small)
            return format(dec.normalize()), dec
    except InvalidOperation:
        return last, None

def is_equal_numeric(a_obj, b_obj, tol=Decimal('1e-6')):
    """Compare numeric objects: Fractions compared exactly; Decimal compared with tolerance."""
    if a_obj is None or b_obj is None:
        return False
    if isinstance(a_obj, Fraction) and isinstance(b_obj, Fraction):
        return a_obj == b_obj
    try:
        a_dec = Decimal(a_obj)
        b_dec = Decimal(b_obj)
        return abs(a_dec - b_dec) <= tol
    except Exception:
        # fallback exact equality of string
        return str(a_obj) == str(b_obj)

def generate_local(model, tokenizer, prompts, device, **gen_kwargs):
    """
    Generate with local HF model.
    If gen_kwargs['num_return_sequences'] > 1 and do_sample True, will produce multiple outputs per prompt in order.
    Returns list of lists: outputs_per_prompt[i] = [str1, str2, ...]
    """
    model.to(device)
    model.eval()
    outputs_all = []
    batch_size = gen_kwargs.get('batch_size', 1)
    # ensure inputs as list
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        enc = tokenizer(batch_prompts, return_tensors='pt', padding=True)
        input_ids = enc['input_ids'].to(device)
        attn = enc['attention_mask'].to(device)
        with torch.no_grad():
            gen_out = model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                **{k:v for k,v in gen_kwargs.items() if k!='batch_size'}
            )
        # gen_out shape: (batch * num_return_sequences, seq_len)
        # decode and split per prompt
        decoded = tokenizer.batch_decode(gen_out, skip_special_tokens=True)
        num_return = gen_kwargs.get('num_return_sequences', 1)
        for b_idx in range(len(batch_prompts)):
            chunk = decoded[b_idx*num_return:(b_idx+1)*num_return]
            outputs_all.append(chunk)
    return outputs_all

def evaluate(args):
    ds = load_dataset("openai/gsm8k",'main', split="test")
    # optionally limit
    if args.max_examples:
        ds = ds.select(range(min(args.max_examples, len(ds))))
    questions = [ex['question'] for ex in ds]
    answers = [ex.get('answer') or ex.get('correct_answer') or ex.get('target') for ex in ds]
    # load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    # For causal LM tokenizer (like GPT-2) ensure padding side and pad token
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token':'<|pad|>'})
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, 
        device_map="auto")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = '<|pad|>'
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    # build prompts
    few_shot_text = None
    if args.few_shot_file:
        with open(args.few_shot_file, 'r', encoding='utf-8') as f:
            few_shot_text = f.read()
    prompts = [build_prompt(q, few_shot_text) for q in questions]

    # generation params
    if args.mode == 'greedy':
        gen_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=False, num_return_sequences=1, batch_size=args.batch_size)
    else:
        gen_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=True, temperature=args.temperature,
                          top_p=args.top_p, num_return_sequences=args.k, batch_size=args.batch_size)
    # run
    raw_outputs = generate_local(model, tokenizer, prompts, device, **gen_kwargs)

    # process outputs, compute per-question predicted answer (for greedy: 1 item; for maj: take majority)
    preds = []
    preds_strs = []
    for i, out_list in enumerate(raw_outputs):
        parsed = []
        str_parsed = []
        for out in out_list:
            # model output includes prompt + continuation sometimes; try to strip prompt
            # safer approach: take text after the prompt's last part "The answer is"
            tail = "The answer is"
            if tail in out:
                cont = out.split(tail)[-1]
            else:
                # fallback: use whole output
                cont = out
            s, numobj = canonicalize_numeric(cont)
            parsed.append((s, numobj))
            str_parsed.append(s)
        # choose final: majority on canonical string (exclude None)
        cand_strings = [p[0] for p in parsed if p[0] is not None]
        if not cand_strings:
            final = (None, None)
        else:
            most = Counter(cand_strings).most_common(1)[0][0]
            # find associated numeric object for comparison (take first matching)
            idx = cand_strings.index(most)
            final = parsed[idx]
        preds.append(final)
        preds_strs.append(str_parsed)

    # canonicalize ground truth answers too
    gt_parsed = []
    for a in answers:
        s, numobj = canonicalize_numeric(a if a else "")
        gt_parsed.append((s, numobj))

    # compute accuracy
    correct = 0
    details = []
    for i, ((pred_s, pred_obj), (gt_s, gt_obj)) in enumerate(zip(preds, gt_parsed)):
        ok = False
        if pred_s is not None and gt_s is not None:
            ok = is_equal_numeric(pred_obj, gt_obj)
        if ok:
            correct += 1
        details.append({
            'index': i,
            'question': questions[i],
            'gt': gt_s,
            'pred': pred_s,
            'pred_all': preds_strs[i],
            'correct': ok
        })
    acc = correct / len(preds)
    print(f"Mode={args.mode} model={args.model_name_or_path} examples={len(preds)} Accuracy={acc:.4f}")
    # write details to file
    outf = args.output if args.output else f"gsm8k_eval_{args.mode}_llama3_8b.jsonl"
    with open(outf, 'w', encoding='utf-8') as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
    print("Detailed results written to", outf)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--mode", choices=['greedy','maj'], default='greedy', help="greedy or maj (self-consistency/maj@k)")
    parser.add_argument("--k", type=int, default=16, help="number of samples for maj@k")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--few_shot_file", type=str, default=None, help="optional file with few-shot examples to prepend")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    evaluate(args)
