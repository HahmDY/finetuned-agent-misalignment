<h1 align="center"><img src="assets/RedCode-logo-512.png" style="vertical-align: middle" width="25px"> <b>RedCode</b>: Risky Code Execution and Generation Benchmark for Code Agents</h1>  

<p align="center">
  📄 <a href="https://arxiv.org/abs/2411.07781"><b>[Paper]</b></a>
  🌐 <a href="https://redcode-agent.github.io"><b>[Webpage]</b></a>
</p>


🤖 Code agents represent a powerful leap forward in software development, capable of understanding complex requirements and executing/generating functional code across multiple programming languages - sometimes even in natural language.


⚠️ But code agents are not risk-free.

In this work, we propose RedCode, a high-quality, large-scale (over 4,000 test cases) dataset that features diverse languages and formats (Python, Bash, natural language), providing real interaction with systems and fine-grained evaluation of code execution and generation, aiming to rigorously and comprehensively evaluate the safety of code agents.
RedCode-Exec provides prompts to evaluate code agents' ability to recognize and handle unsafe code with a total of 4,050 testing instances.

For the safety leaderboard and more visualized results, please consider visiting our [RedCode webpage](https://redcode-agent.github.io).

## 📂 Repository Structure

### Dataset

This directory contains the datasets `RedCode-Exec`, which are used as inputs for the agents.

### Environment

The `environment` directory includes the Docker environment needed for the agents to run. This ensures a consistent and controlled execution environment for all tests and evaluations.

### Evaluation

The `evaluation` directory contains subdirectories for the evaluation.
Additionally, `evaluation.py` that serve as evaluation scripts for each risky scenario.

### Result

The `result` directory stores the results of the evaluations.

### Scripts

The `scripts` directory contains the bash scripts to run the evaluations for CA agent.

## 🎯 Getting Started

Follow these steps to set up the project locally.

### Environment Setup

The `environment.yml` file lists all dependencies required for the project. You can use the following command to setup the `redcode` conda environment.

```bash
conda env create -f environment.yml
conda activate redcode
pip install -e .
```

### 📝 Evaluation

#### Model Deployment

To run the evaluation, you first need to deploy your model using vLLM. If you have a fine-tuned model, please refer to the [Fine-tuning Guide](/LLaMA-Factory/README_ft.md) for instructions. Use the following commands to deploy your model:

- **Deploying a base LLM:**

```bash
python -m vllm.entrypoints.openai.api_server \
    --model "meta-llama/Llama-3.1-8B-Instruct" \
    --served-model-name llama_it \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code
```

- **Deploying a fine-tuned agent:**

```bash
python -m vllm.entrypoints.openai.api_server \
    --model "$PING_HOME/Models/code_llama" \
    --served-model-name code_llama \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code
```

> **Note:** The `--trust-remote-code` option is only required for models based on GLM.

#### Running the Benchmark

You can evaluate the safety of your code agent using the `scripts/run_agent.sh` script. The script takes the following parameters:

- `model`: The name used when deploying the model with vLLM
- `safety_system`: Whether to use a safe system prompt (include this option to enable PTST)
- `prefix`: The prefix to use for prefix injection

### Guard

The `/guard` folder provides the baseline Guard code. Llama Guard is used to check whether each instruction is refused.

To use this, first deploy the Llama Guard model with vLLM:

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

Our code is heavily based off the [RedCode codebase](https://github.com/AI-secure/RedCode).
```
@article{guo2024redcode,
  title={RedCode: Risky Code Execution and Generation Benchmark for Code Agents},
  author={Guo, Chengquan and Liu, Xun and Xie, Chulin and Zhou, Andy and Zeng, Yi and Lin, Zinan and Song, Dawn and Li, Bo},
  booktitle={Thirty-Eighth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2024}
}
```