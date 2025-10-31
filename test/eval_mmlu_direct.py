import argparse
import os
import torch
import numpy as np
import pandas as pd
from categories import subcategories, categories
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

choices = ["A", "B", "C", "D"]


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
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


@torch.no_grad()
def eval(args, subject, model, tokenizer, dev_df, test_df, device):
    cors = []
    all_probs = []

    max_len = getattr(tokenizer, "model_max_length", 2048)

    for i in range(test_df.shape[0]):
        # build few-shot prompt
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        # encode and move to device; ensure attention_mask present
        enc = tokenizer(prompt, return_tensors="pt", truncation=False)
        input_ids = enc.input_ids.to(device)
        attention_mask = enc.attention_mask.to(device) if "attention_mask" in enc else torch.ones_like(input_ids, device=device)

        # reduce k if too long
        while input_ids.shape[-1] > max_len and k > 0:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            enc = tokenizer(prompt, return_tensors="pt", truncation=False)
            input_ids = enc.input_ids.to(device)
            attention_mask = enc.attention_mask.to(device) if "attention_mask" in enc else torch.ones_like(input_ids, device=device)

        # label normalization
        label = test_df.iloc[i, test_df.shape[1] - 1]
        if isinstance(label, (int, np.integer)):
            label = choices[int(label)]
        label = str(label).strip()

        # forward (causal LM): pass attention_mask explicitly
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (1, seq_len, vocab)
        last_logits = logits[0, -1, :]  # (vocab,)

        # map choices to token ids (try both "A" and " A")
        token_ids = []
        for ch in choices:
            ids_with_space = tokenizer(" " + ch, return_tensors="pt").input_ids[0].tolist()
            ids_no_space = tokenizer(ch, return_tensors="pt").input_ids[0].tolist()
            chosen = None
            # prefer single-token encodings; prefer leading-space encoding if single token
            if len(ids_with_space) == 1:
                chosen = ids_with_space[0]
            elif len(ids_no_space) == 1:
                chosen = ids_no_space[0]
            else:
                chosen = None
            token_ids.append(chosen)

        # if all choices map to single token, compute probs directly
        if all(tid is not None for tid in token_ids):
            tid_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)
            choice_logits = last_logits[tid_tensor]  # (num_choices,)
            probs = torch.nn.functional.softmax(choice_logits, dim=0).detach().cpu().numpy()
            pred = choices[int(np.argmax(probs))]
        else:
            # fallback: use generate (1 token) with explicit attention_mask and pad/eos ids
            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
                # temperature=0.0,
            )
            out = tokenizer.decode(gen[0], skip_special_tokens=True).strip()
            # try to match A/B/C/D in generated string
            found = None
            for ch in choices:
                if out == ch or out.endswith(ch) or out.startswith(ch) or (" " + ch) in out:
                    found = ch
                    break
            pred = found if found is not None else out
            # cannot compute reliable per-choice probs in fallback
            probs = np.full((len(choices),), np.nan)

        cor = (pred == label)
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)
    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs


def main(args):
    # choose device (single specified GPU)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
        print("CUDA not available; running on CPU (very slow for Llama3-8B)")

    print("Loading tokenizer and model from:", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    # --- Ensure pad token and model.config.pad_token_id are set to avoid warnings ---
    if tokenizer.pad_token is None:
        # Lightweight approach: set pad token to eos token (does not change vocab size)
        tokenizer.pad_token = tokenizer.eos_token
    # Configure model after load, but set tokenizer pad token first.
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)
    # Sync pad_token_id & eos_token_id into model config to suppress warnings
    if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model.config, "eos_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        model.config.eos_token_id = tokenizer.eos_token_id
    # move model to specified device
    model.to(device)
    model.eval()
    # ------------------------------------------------------------------------------

    # build subject list from test directory
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    # prepare save directories
    model_name = os.path.basename(os.path.normpath(args.model))
    save_results_dir = os.path.join(args.save_dir, f"direct_results_{model_name}")
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(save_results_dir):
        os.makedirs(save_results_dir)

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

    for subject in subjects:
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
        )[: args.ntrain]
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        )

        cors, acc, probs = eval(args, subject, model, tokenizer, dev_df, test_df, device)
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        # add result columns and save
        test_df[f"{model_name}_correct"] = cors
        # if probs is shape (n_samples, n_choices), add columns; if fallback had nan it's still written
        if probs.ndim == 2 and probs.shape[0] == test_df.shape[0]:
            for j in range(probs.shape[1]):
                choice = choices[j]
                test_df[f"{model_name}_choice{choice}_probs"] = probs[:, j]
        else:
            # fallback case: fill NaNs
            for j in range(len(choices)):
                choice = choices[j]
                test_df[f"{model_name}_choice{choice}_probs"] = np.nan

        out_path = os.path.join(save_results_dir, f"{subject}.csv")
        test_df.to_csv(out_path, index=None)
        print(f"Saved results to {out_path}")

    # aggregate and print per-subcategory and per-category accuracies
    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--gpu", "-G", type=int, default=2, help="要使用的 GPU 索引，例如 0")
    parser.add_argument("--data_dir", "-d", type=str, default="./datasets/mmlu/data", help="数据目录")
    parser.add_argument("--save_dir", "-s", type=str, default="./results/mmlu", help="结果保存目录")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="./Llama3-8B",  # 请替换为你本地模型目录
    )
    args = parser.parse_args()
    main(args)
