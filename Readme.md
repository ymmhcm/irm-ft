MMLU测评 vllm ver
step1:
(/home/game/disk_sdb/conda_envs/irm-ft) game@game-4U-GPU-Server:~/disk_sdb/workspace_ymm$ bash vllm_run.sh
step2:
(/home/game/disk_sdb/conda_envs/irm-ft) (.venv) game@game-4U-GPU-Server:~/disk_sdb/workspace_ymm$ python test/eval_mmlu_vllm.py --model Llama3-8B

MMLU测评 direct ver
(/home/game/disk_sdb/conda_envs/irm-ft) game@game-4U-GPU-Server:~/disk_sdb/workspace_ymm$ python test/eval_mmlu_direct.py