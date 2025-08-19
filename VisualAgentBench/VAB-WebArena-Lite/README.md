# Setup for VAB-WebArena-Lite

## Table of Contents

- [Brief Introduction](#brief-introduction)
- [Install](#install)
- [Setup WebArena-Lite Environments](#setup-webarena-lite-environments)
- [🚨 Important: Refresh all websites before re-run another round of testing](#-important-refresh-all-websites-before-re-run-another-round-of-testing)
- [🚀 Evaluating in WebRL Setting (Text Modal)](#-evaluating-in-webrl-setting-text-modal)
  - [Evaluation of Finetuned Models](#evaluation-of-finetuned-models)
  - [Evaluation of Proprietary Models](#evaluation-of-proprietary-models)
- [Run Visualized Demostration](#run-visualized-demostration)
- [Acknowledgements](#acknowledgements)

## Brief Introduction

VAB-WebArena-Lite is a 165-task refined subset from <a href="https://webarena.dev/" target="_blank">WebArena</a>.
The purpose of building this subset is to manually ensure task correctness & feasibility, and speed up testing (original 812-task WebArena usually takes more than 6h to run through, while VAB-WebArena-Lite takes around 40m in practice). 
The modified version of the test cases can be found in `config_files/wa/test_webarena_lite.raw.json`.


## Install

First, you should install the dependencies for VAB-WebArena-Lite (recommend using an independent conda environment to VAB):

```bash
conda create -n webarena python=3.10
conda activate webarena
pip install -r requirements.txt
playwright install
pip install -e .
```

You can also run the unit tests to ensure that WebArena-Lite is installed correctly:

```bash
pytest -x
```

## Setup WebArena-Lite Environments

1. Setup the standalone environments.
Please check out [this page](https://github.com/web-arena-x/webarena/tree/main/environment_docker) for details.

2. Configurate the urls for each website.
First, export the `DATASET` to be `webarena`:

```bash
export DATASET=webarena
```

Then, set the URL for the websites
(🚨 Notice: check if default ports of websites below correspond to those you setup in the first step)

```bash
# Actually, the CLASSIFIEDS environment is not included in the WebArena-Lite evaluation, we keep the environment variables here just for consistency.
export CLASSIFIEDS="<your_classifieds_domain>:9980"
export CLASSIFIEDS_RESET_TOKEN="4b61655535e7ed388f0d40a93600254c"

# Below are the variables you should set for the evaluation.
export SHOPPING="<your_shopping_site_domain>:7770"
export REDDIT="<your_reddit_domain>:9999"
export SHOPPING_ADMIN="<your_e_commerce_cms_domain>:7780/admin"
export GITLAB="<your_gitlab_domain>:8023"
export MAP="<your_map_domain>:3000"
export WIKIPEDIA="<your_wikipedia_domain>:8888"
export HOMEPAGE="<your_homepage_domain>:4399"
```

3. Generate config files for each test example:

```bash
python scripts/generate_test_data.py
```

You will see `*.json` files generated in the [config_files](./config_files) folder. Each file contains the configuration for one test example.

4. Obtain and save the auto-login cookies for all websites:

```bash
bash prepare.sh
```

5. Set up API keys.

```bash
export OPENAI_API_KEY=your_key

# Optional: if you use a different OpenAI model source
export OPENAI_API_URL=your_url 

# Optional: you can set the following variables to evaluate the preset model in llms/providers/api_utils.py
export GEMENI_API_KEY=your_key
export QWEN_API_KEY=your_key
export CLAUDE_API_KEY=your_key

# Optional: if you have trained your model, we recommend deploying it as an API service, where you can set a FINETUNED_URL to evaluate it.
export FINETUNED_URL=your_url

```

## 🚨 Important: Refresh all websites before re-run another round of testing!
Since tasks in WebArena may involve changing status and database of websites (e.g., posting comments on Reddit), if websites are not all refreshed before another round of evaluation, the results would be problematic.

Please remember to always run following command (assume you are hosting WebArena websites on your own) to restart and refresh all website dockers to avoid potential contamination.
The process usually takes 3-5 minites.

```bash
# Make sure the script is executed on the machine that you run those website dockers
bash refresh_website_docker.sh
```

You may need to change some contents in the script (e.g. configured ports of websites, names of dockers, etc.).


## 📝 Evaluating

This setting is based on [WebRL](https://github.com/THUDM/WebRL). You can use this setting to reproduce the results of our work.

**Before running, remember to read and follow the above environmental setup procedures!**

### Model Deployment

You first need to use tools like vllm to deploy the finetuned model locally. Once deployed, the model can be accessed through the OpenAI API call method. 
Deploy the model as shown below:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model "$PING_HOME/Models/web_llama" \
    --served-model-name web_llama \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code
```

> **Note:** The `--trust-remote-code` option is only required for models based on GLM.

### Running the Benchmark

We provide a parallel script. Use the script below for evaluation:

```bash
# Remember to first launch a tmux session
tmux
bash wa_parallel_run_webrl.sh
```

To use PTST or PING, set the following parameters:
- `safety_system`: If this tag is included, the safe version system prompt will be used.
- `prefix`: Enter the prefix you want to use.
- `model`: The name of the model deployed with vllm.
- `planner_ip`: The URL of the model deployed with vllm. Example: 'http://localhost:8000/v1'

If you want to use the base LLM instead of a fine-tuned agent, use the script below:
```bash
# Remember to first launch a tmux session
tmux
bash wa_parallel_run_webrl_chat.sh
```
### Guard

The `/guard` folder provides the baseline Guard code. Llama Guard is used to check whether each instruction is refused.
Since an initial observation is required, please first perform the evaluation using the fine-tuned Llama, and then follow the steps below.

First, deploy the Llama Guard model with vLLM:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model "meta-llama/Llama-Guard-3-8B" \
    --served-model-name llama_guard \
    --host 0.0.0.0 \
    --port 8000
```

Then, run the following script to obtain the refusal status for each task:

```bash
python guard/guard.py
```

## Acknowledgements

Please cite our paper if you find VAB-WebArena-Lite useful for your work:



Our code is heavily based off the <a href="https://github.com/THUDM/VisualAgentBench">VisualAgentBench codebase</a>, <a href="https://github.com/web-arena-x/webarena">WebArena codebase</a> and <a href="https://github.com/web-arena-x/visualwebarena">VisualWebArena codebase</a>.

```bibtex
@article{liu2024visualagentbench,
  title={VisualAgentBench: Towards Large Multimodal Models as Visual Foundation Agents},
  author={Liu, Xiao and Zhang, Tianjie and Gu, Yu and Iong, Iat Long and Xu, Yifan and Song, Xixuan and Zhang, Shudan and Lai, Hanyu and Liu, Xinyi and Zhao, Hanlin and others},
  journal={arXiv preprint arXiv:2408.06327},
  year={2024}
}

@article{koh2024visualwebarena,
  title={VisualWebArena: Evaluating Multimodal Agents on Realistic Visual Web Tasks},
  author={Koh, Jing Yu and Lo, Robert and Jang, Lawrence and Duvvur, Vikram and Lim, Ming Chong and Huang, Po-Yu and Neubig, Graham and Zhou, Shuyan and Salakhutdinov, Ruslan and Fried, Daniel},
  journal={arXiv preprint arXiv:2401.13649},
  year={2024}
}

@article{zhou2024webarena,
  title={WebArena: A Realistic Web Environment for Building Autonomous Agents},
  author={Zhou, Shuyan and Xu, Frank F and Zhu, Hao and Zhou, Xuhui and Lo, Robert and Sridhar, Abishek and Cheng, Xianyi and Bisk, Yonatan and Fried, Daniel and Alon, Uri and others},
  journal={ICLR},
  year={2024}
}
```

