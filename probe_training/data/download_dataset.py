# download_datasets.py

import os
import requests
import pandas as pd
import json

def download_text(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.text

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def main():
    # Alpaca
    alpaca_url = "https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/main/alpaca_data_cleaned.json"
    alpaca_path = "alpaca_prompts.json"
    if not os.path.exists(alpaca_path):
        print("Downloading Alpaca...")
        alpaca_json = json.loads(download_text(alpaca_url))
        prompts = []
        for x in alpaca_json:
            instr = x.get("instruction", "").strip()
            inp = x.get("input", "").strip()
            if instr:
                if inp:
                    full_prompt = f"{instr}: {inp}"
                else:
                    full_prompt = instr
                prompts.append({"prompt": full_prompt})
        save_json(prompts, alpaca_path)
        print(f"Saved: {alpaca_path}")

    # HarmBench (.csv → .json)
    harmbench_csv_url = "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv"
    harmbench_json_path = "harmbench_prompts.json"
    if not os.path.exists(harmbench_json_path):
        print("Downloading HarmBench...")
        df = pd.read_csv(harmbench_csv_url)
        prompts = [{"prompt": x} for x in df["Behavior"].dropna().tolist()]
        save_json(prompts, harmbench_json_path)
        print(f"Saved: {harmbench_json_path}")

    # AdvBench (.csv → .json)
    advbench_csv_url = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/advbench/harmful_behaviors.csv"
    advbench_json_path = "advbench_prompts.json"
    if not os.path.exists(advbench_json_path):
        print("Downloading AdvBench...")
        df = pd.read_csv(advbench_csv_url)
        prompts = [{"prompt": x} for x in df["goal"].dropna().tolist()]
        save_json(prompts, advbench_json_path)
        print(f"Saved: {advbench_json_path}")

if __name__ == "__main__":
    main()
