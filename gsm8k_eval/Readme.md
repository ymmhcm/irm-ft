gsm8k数据集在hf的默认下载路径里
local_hf.py是直接用hf加载权重进行gsm8k推理，结果在gsm8k_eval_greedy.jsonl和其上一个文件里，但是结果很差，只有0.2几，应该是没跑对，这不符合leaderbord里面结果

output文件夹里是evalscope的结果
evalscope eval --model /home/game/disk_sdb/workspace_ymm/catastrophic_forgetting/Llama3-8B --datasets gsm8k

causal_cot_gsm8k.py是实现的最简单的基于个人因果效应估计实现的存储训练集里的因果cot用于训练集
python causal_cot_gsm8k.py
报告了baseline和methed 记忆存在了ChromaDB里，用的llamafactory cli进行vllm本地部署 8001

comparative_experiment.py是对比我们的方法和navie rag
用的llamafactory cli进行vllm本地部署 8002

2025.11.22
使用的检索方式是"Question" 去匹配 "Rule" 按道理来说 应该还有需要更好的匹配方式
python causal_cot_gsm8k.py 200训练集合，50测试集合
==================== Baseline: Standard Zero-shot CoT ====================
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [05:09<00:00,  6.18s/it]

[Baseline Result] Accuracy: 18/50 = 36.00%

==================== Phase 1: Causal Memory Mining ====================
Mining: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [23:56<00:00,  7.18s/it]

[Mining Done] Saved 82 causal rules from 200 samples.

==================== Phase 2: Inference with Causal Memory ====================
[RAG] SUCCESS! Fixed a hard problem: Janet’s ducks lay 16 eggs per ...
[RAG] SUCCESS! Fixed a hard problem: A robe takes 2 bolts of blue f...
[RAG] SUCCESS! Fixed a hard problem: Toulouse has twice as many she...
[RAG] SUCCESS! Fixed a hard problem: Jill gets paid $20 per hour to...
[RAG] SUCCESS! Fixed a hard problem: Claire makes a 3 egg omelet ev...
[RAG] SUCCESS! Fixed a hard problem: I have 10 liters of orange dri...
[RAG] SUCCESS! Fixed a hard problem: Cynthia eats one serving of ic...
[RAG] SUCCESS! Fixed a hard problem: Mike plays ping pong for 40 mi...
[RAG] SUCCESS! Fixed a hard problem: Terry eats 2 yogurts a day.  T...
[RAG] SUCCESS! Fixed a hard problem: Dana can run at a rate of spee...
[RAG] SUCCESS! Fixed a hard problem: Brandon's iPhone is four times...
[RAG] SUCCESS! Fixed a hard problem: Tracy used a piece of wire 4 f...
[RAG] SUCCESS! Fixed a hard problem: Richard lives in an apartment ...

==============================
FINAL REPORT
==============================
Baseline Accuracy: 36.00%
Ours (Causal) Acc: 42.00%
Improvement:       +6.00%
Fixed Hard Cases:  13 (Problems baseline failed but we solved)
==============================