gsm8k数据集在hf的默认下载路径里
local_hf.py是直接用hf加载权重进行gsm8k推理，结果在gsm8k_eval_greedy.jsonl和其上一个文件里，但是结果很差，只有0.2几，应该是没跑对，这不符合leaderbord里面结果

output文件夹里是evalscope的结果 4 shot，见/home/game/disk_sdb/workspace_ymm/gsm8k_eval/outputs/20251122_125022/configs/task_config_e28e3e.yaml
evalscope eval --model /home/game/disk_sdb/workspace_ymm/catastrophic_forgetting/Llama3-8B --datasets gsm8k
+-----------+-----------+----------+----------+-------+---------+---------+
| Model     | Dataset   | Metric   | Subset   |   Num |   Score | Cat.0   |
+===========+===========+==========+==========+=======+=========+=========+
| Llama3-8B | gsm8k     | mean_acc | main     |  1319 |  0.7862 | default |
+-----------+-----------+----------+----------+-------+---------+---------+ 

causal_cot_gsm8k.py是实现的最简单的基于个人因果效应估计实现的存储训练集里的因果cot用于训练集
python causal_cot_gsm8k.py
报告了baseline和methed 记忆存在了ChromaDB里，用的llamafactory cli进行vllm本地部署 8001

comparative_experiment.py是对比我们的方法和navie rag
用的llamafactory cli进行vllm本地部署 8002

step1 
运行llama factory
(/home/game/disk_sdb/conda_envs/irm-ft) game@game-4U-GPU-Server:~/disk_sdb/workspace_ymm/LLaMA-Factory$ bash run.sh

step2
运行测试文件
(/home/game/disk_sdb/conda_envs/irm-ft) game@game-4U-GPU-Server:~/disk_sdb/workspace_ymm/gsm8k_eval$ python unified_gsm8k.py

2025.11.22
使用的检索方式是"Question" 去匹配 "Rule" 按道理来说 应该还有需要更好的匹配方式
python causal_cot_gsm8k.py 200训练集合，50测试集合 tem=0.7
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

python causal_cot_gsm8k.py 200训练集合，50测试集合 tem=0.7
Loading GSM8K dataset...

==================== Baseline: Standard Zero-shot CoT ====================
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [05:16<00:00,  6.33s/it]

[Baseline Result] Accuracy: 23/50 = 46.00%

==================== Phase 1: Causal Memory Mining ====================
Mining: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [25:34<00:00,  7.67s/it]

[Mining Done] Saved 103 causal rules from 200 samples.

==================== Phase 2: Inference with Causal Memory ====================
==============================
FINAL REPORT
==============================
Baseline Accuracy: 46.00%
Ours (Causal) Acc: 60.00%
Improvement:       +14.00%
Fixed Hard Cases:  15 (Problems baseline failed but we solved)
==============================

Training: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [56:21<00:00, 16.91s/it]

==============================
Training Finished.
Skipped (Already Cached): 0
Newly Processed:          153
Current Naive Count:      153
Current Causal Count:     139
==============================

python comparative_experiment.py 200训练集合，50测试集合 tem=0.7

开始对比测试 (Test Size: 50)...
Testing: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [25:20<00:00, 30.42s/it]

########################################
FINAL COMPARISON REPORT
########################################
Total Test Samples: 50
Naive RAG Accuracy:  46.00%
Causal-CoT Accuracy: 42.00%
Delta (Ours - Naive): -4.00%
----------------------------------------
CONCLUSION: Causal-CoT underperformed. Check if the causal filter is too strict.

2025.11.23
python causal_cot_gsm8k.py 200训练集合，1319测试集合 tem=0.7
[Mining Done] Saved 83 causal rules from 200 samples.
==============================
FINAL REPORT
==============================
Baseline Accuracy: 45.79%
Ours (Causal) Acc: 55.34%
Improvement:       +9.55%
Fixed Hard Cases:  358 (Problems baseline failed but we solved)
==============================

python comparative_experiment.py 200训练集合，1319测试集合 tem=0.7
==============================
Training Finished.
Skipped (Already Cached): 0
Newly Processed:          160
Current Naive Count:      160
Current Causal Count:     146
==============================

########################################
FINAL COMPARISON REPORT
########################################
Total Test Samples: 1319
Naive RAG Accuracy:  58.30%
Causal-CoT Accuracy: 54.89%
Delta (Ours - Naive): -3.41%
----------------------------------------
CONCLUSION: Causal-CoT underperformed. Check if the causal filter is too strict.

2025.11.23
python causal_cot_gsm8k.py 600训练集合，1319测试集合 tem=0.7
==================== Phase 1: Causal Memory Mining ====================
Mining: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 600/600 [1:11:41<00:00,  7.17s/it]

[Mining Done] Saved 254 causal rules from 600 samples.

==================== Phase 2: Inference with Causal Memory ====================

==============================
FINAL REPORT
==============================
Baseline Accuracy: 45.79%
Ours (Causal) Acc: 53.15%
Improvement:       +7.35%
Fixed Hard Cases:  339 (Problems baseline failed but we solved)
==============================

python comparative_experiment.py 600训练集合，1319测试集合 tem=0.7

Training: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 600/600 [3:06:47<00:00, 18.68s/it]

==============================
Training Finished.
Skipped (Already Cached): 0
Newly Processed:          478
Current Naive Count:      478
Current Causal Count:     441
==============================

开始对比测试 (Test Size: 1319)...
Testing: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 1319/1319 [9:25:27<00:00, 25.72s/it]

########################################
FINAL COMPARISON REPORT
########################################
Total Test Samples: 1319
Naive RAG Accuracy:  52.69%
Causal-CoT Accuracy: 53.98%
Delta (Ours - Naive): +1.29%
----------------------------------------
CONCLUSION: Causal-CoT achieved similar/better performance with MUCH smaller memory size.
This proves that filtering out spurious correlations reduces retrieval noise.


python unified_gsm8k.py 1000训练集合，1319测试集合 tem=0.7
