import argparse
import os
import numpy as np
import pandas as pd
import time
import requests  # 用于调用vllm的API

from crop import crop

# 配置本地vllm服务地址
VLLM_API_URL = "http://localhost:8000/v1/completions"
choices = ["A", "B", "C", "D"]


def softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator/denominator
    return softmax

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt

def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

def eval(args, subject, model_name, dev_df, test_df):
    cors = []
    all_probs = []
    answers = choices[:test_df.shape[1]-2]

    for i in range(test_df.shape[0]):
        # 获取提示并确保其适合模型输入
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        # 如果提示过长则减少训练示例数量
        while crop(prompt) != prompt:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end

        label = test_df.iloc[i, test_df.shape[1]-1]

        # 调用vllm API获取结果
        while True:
            # print(model_name)
            # print(f"当前测评 {subject} - 第{i+1}题")
            try:
                # 构造vllm请求参数
                payload = {
                    "prompt": prompt,
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "logprobs": 5,  # 关键修改：从100降至5（或50以内）
                    "echo": True,
                    "model": model_name  # 模型名称（与vllm部署的一致，即models/Llama3-8B）
                }
                
                response = requests.post(VLLM_API_URL, json=payload)
                response.raise_for_status()  # 检查请求是否成功
                c = response.json()
                break
            except Exception as e:
                print(f"请求出错: {e}，暂停后重试")
                time.sleep(1)
                continue

        lprobs = []
        for ans in answers:
            try:
                # 提取每个选项的对数概率
                # vllm的logprobs结构与OpenAI类似，但需要确认具体路径
                lprobs.append(c["choices"][0]["logprobs"]["top_logprobs"][-1][" {}".format(ans)])
            except:
                print(f"警告: 未找到选项 {ans} 的概率，使用默认值 -100")
                lprobs.append(-100)
        
        # 计算预测结果
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(lprobs)]
        probs = softmax(np.array(lprobs))

        # 记录是否正确
        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)
    all_probs = np.array(all_probs)
    print(f"{subject}的平均准确率 {acc:.3f} - {subject}")

    return cors, acc, all_probs

def main(args):
    model_names = args.model  # 改用model参数
    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(args.data_dir, "test")) if "_test.csv" in f])

    # 创建保存结果的目录
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    for model in model_names:
        model_dir = os.path.join(args.save_dir, f"results_{model}")
        # 自动创建所有父目录（如果不存在）
        os.makedirs(model_dir, exist_ok=True)
        # if not os.path.exists(model_dir):
        #     os.mkdir(model_dir)

    print("测评科目:", subjects)
    print("参数:", args)

    for model in model_names:
        print(f"正在测评模型: {model}")
        all_cors = []

        for subject in subjects:
            dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", f"{subject}_dev.csv"), header=None)[:args.ntrain]
            test_df = pd.read_csv(os.path.join(args.data_dir, "test", f"{subject}_test.csv"), header=None)

            cors, acc, probs = eval(args, subject, model, dev_df, test_df)
            all_cors.append(cors)

            # 保存详细结果
            test_df[f"{model}_correct"] = cors
            for j in range(probs.shape[1]):
                choice = choices[j]
                test_df[f"{model}_choice{choice}_probs"] = probs[:, j]
            test_df.to_csv(os.path.join(args.save_dir, f"results_{model}", f"{subject}.csv"), index=None)

        # 计算总体平均准确率
        weighted_acc = np.mean(np.concatenate(all_cors))
        print(f"总体平均准确率: {weighted_acc:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5, help="训练示例数量")
    parser.add_argument("--data_dir", "-d", type=str, default="./datasets/mmlu/data", help="数据目录")
    parser.add_argument("--save_dir", "-s", type=str, default="./results/mmlu", help="结果保存目录")
    parser.add_argument("--model", "-m", type=str,nargs="+", required=True, 
                        help="本地vllm部署的模型名称（多个模型用空格分隔）")
    args = parser.parse_args()
    main(args)