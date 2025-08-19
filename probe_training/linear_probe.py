

import torch
from transformer_lens import HookedTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, log_loss
from transformers import AutoTokenizer
import numpy as np
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
import pickle
from torch.nn.utils.rnn import pad_sequence
os.environ["DISABLE_FLASH_ATTENTION"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

def load_prompts(path):
    with open(path, "r") as f:
        data = json.load(f)
    return [item["prompt"] for item in data if "prompt" in item]

def load_goal(path):
    with open(path, "r") as f:
        data = json.load(f)
    return [item["goal"] for item in data if "goal" in item]

alpaca = load_prompts("data/alpaca_prompts.json")
harmbench = load_prompts("data/harmbench_prompts.json")
advbench = load_prompts("data/advbench_prompts.json")

webdata_benign_goal = load_goal("data/webagent_input_benign.json")
webdata_harmful_goal = load_goal("data/webagent_input_harmful.json")

# harmful train / test set
harmbench_train = harmbench
advbench_train = advbench
harmful_train = harmbench_train + advbench_train
harmful_test = webdata_harmful_goal

# safe train / test set
safe_train = alpaca[:920]
safe_test = webdata_benign_goal

train_prompts = harmful_train + safe_train
train_labels = [1] * len(harmful_train) + [0] * len(safe_train)

test_prompts = harmful_test + safe_test
test_labels = [1] * len(harmful_test) + [0] * len(safe_test)

print(f"Train size: {len(train_prompts)} (harm: {len(harmful_train)}, safe: {len(safe_train)})")
print(f"Test size: {len(test_prompts)} (harm: {len(harmful_test)}, safe: {len(safe_test)})")

def load_model(path):
    print(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        device_map=None,
        trust_remote_code=True
    )
    device = "cuda:0"
    model.to(device)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def get_template_function(model_tag):
    if "llama" in model_tag.lower():
        def template(prompt):
            return (
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
                f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            )
    elif "qwen" in model_tag.lower():
        def template(prompt):
            return (
                f"<|im_start|>user\n{prompt}\n<|im_end|><|im_start|>assistant"
            )
    elif "glm" in model_tag.lower():
        def template(prompt):
            return (
                f"[gMASK]<sop><|user|>\n{prompt}\n<|assistant|>"
            )
    else:
        raise ValueError(f"Unknown model_tag format: {model_tag}")
    
    return template


def get_activation(model, prompts, target_layer_idxs, template, batch_size=32):
    pooled_outputs_per_layer = {layer_idx: [] for layer_idx in target_layer_idxs}

    formatted_outputs = []

    for i in tqdm(range(0, len(prompts), batch_size), desc=f"Getting activations from layer {target_layer_idxs}"):
        batch = prompts[i:i+batch_size]

        full_ids_list = []
        lengths_list = []
        with torch.no_grad():
            for p in batch:
                chat_prompt = template(p)
                tokens = tokenizer(
                    chat_prompt,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=2048,
                    add_special_tokens=False
                )
                input_ids = tokens["input_ids"].to(model.device)

                full_ids_list.append(input_ids[0])
                lengths_list.append(input_ids.shape[1])

                decoded_str = tokenizer.decode(input_ids[0], skip_special_tokens=False)
                formatted_outputs.append(decoded_str)

            full_ids = pad_sequence(full_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id).to(model.device)
            attention_mask = (full_ids != tokenizer.pad_token_id).long()

        with torch.no_grad():
            outputs = model(
                input_ids=full_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            hidden_states = outputs.hidden_states
            for layer_idx in target_layer_idxs:
                acts = hidden_states[layer_idx]

                batch_hidden = []
                for b in range(len(full_ids_list)):
                    start_idx = 0
                    end_idx = lengths_list[b]
                    print(f"[Layer {layer_idx}] [Seq {b}] Averaging from token {start_idx} to {end_idx - 1}")

                    avg_activation = acts[b, start_idx:end_idx].mean(dim=0)
                    batch_hidden.append(avg_activation)

                batch_hidden = torch.stack(batch_hidden, dim=0)
                pooled_outputs_per_layer[layer_idx].append(batch_hidden.float().cpu())

        
    for layer_idx in target_layer_idxs:
        pooled_outputs_per_layer[layer_idx] = torch.cat(pooled_outputs_per_layer[layer_idx], dim=0).numpy()


    return pooled_outputs_per_layer, formatted_outputs


def save_classification_outputs(prompts, labels, formatted_prompts, save_path, logits):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    results = []
    for i in range(len(prompts)):
        results.append({
            "input": formatted_prompts[i],
            "true_label": int(labels[i]),
            "logit": float(logits[i])
        })
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)


# Qwen/Qwen2.5-7B-Instruct
# THUDM/glm-4-9b-chat
# meta-llama/Meta-Llama-3.1-8B-Instruct

if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--model_tag", type=str, default="llama")
    args = parser.parse_args()

    model, tokenizer = load_model(path=args.model_path)

    template = get_template_function(args.model_tag)

    probe_dir = f"probe/probe_{args.model_tag}/"
    log_dir = f"logs/{args.model_tag}_log/"

    os.makedirs(probe_dir, exist_ok=True)

    target_layers = [20, 22, 24]

    X_train_dict, formatted_train = get_activation(
        model,
        train_prompts,
        target_layer_idxs=target_layers,
        template = template,
        batch_size=32
    )

    y_train = np.array(train_labels)
    clf_train_dict = {}

    os.makedirs(probe_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    for layer, X_train in X_train_dict.items():
        print(f"\n==== Training: Layer {layer} ====")

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        clf_train_dict[layer] = clf

        with open(f"{probe_dir}/layer_{layer}.pkl", "wb") as f:
            pickle.dump(clf, f)

        logits_train = clf.decision_function(X_train)

        save_classification_outputs(
            prompts=train_prompts,
            labels=y_train,
            formatted_prompts=formatted_train,
            save_path=f"{log_dir}/layer_{layer}_train.json",
            logits=logits_train
        )

    X_web_dict, formatted_web = get_activation(
        model,
        test_prompts,
        target_layer_idxs=target_layers,
        template=template,
        batch_size=32
    )


    y_test = np.array(test_labels)

    for layer in target_layers:
        print(f"\n==== Layer {layer} ====")

        with open(f"{probe_dir}/layer_{layer}.pkl", "rb") as f:
            clf = pickle.load(f)

        X_test = X_web_dict[layer]
        logits_test = clf.decision_function(X_test)

        save_classification_outputs(
            prompts=test_prompts,
            labels=y_test,
            formatted_prompts=formatted_web,
            save_path=f"{log_dir}/layer_{layer}.json",
            logits=logits_test
        )

