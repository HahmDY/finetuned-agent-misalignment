# Fine-Tuning LLM Agents with LLaMA-Factory

This guide provides step-by-step instructions to reproduce our fine-tuning process for LLM agents using [LLaMA-Factory](https://github.com/your-org/LLaMA-Factory). We fine-tune models with agentic datasets for web navigation and code generation tasks.

## 🚀 Step-by-Step Instructions

### 1. Set Up the Conda Environment

Create and activate a conda environment with the required Python version, then install the package:

```bash
conda create -n llamaenv python=3.10 -y
conda activate llamaenv
pip install -e .
```

### 2. Download Datasets

Download the datasets used for training:

- **Web Navigation**: [web_data](https://github.com/THUDM/WebRL/blob/main/LLaMA-Factory/data/web_policy_sft.json)
- **Code Generation**: [code_data](https://huggingface.co/datasets/xingyaoww/code-act) (from HuggingFace)

After downloading, save them to the following paths:

```
LLaMA-Factory/data/web_raw.json
LLaMA-Factory/data/code_raw.json
```

### 3. Process the Data

Run the following script to convert the raw JSON files into a format suitable for training:

```bash
python LLaMA-Factory/data/data_convert.py
```

### 4. Fine-Tune the Model

Use the following command to fine-tune the model. Replace `<domain>` with either `web` or `code`, and `<model>` with one of `llama`, `glm`, or `qwen`.

```bash
bash agent_finetune.sh <domain> <model>
```

**Example:**

```bash
bash agent_finetune.sh web llama
```

> **Note:**
> The fine-tuned model will be saved in the following directory format:
> 
> ```
> LLaMA-Factory/Models/{domain}_{base_model}
> ```
> 
> For example, if you fine-tune the web domain with the llama base model, the output will be saved at:
> 
> ```
> LLaMA-Factory/Models/web_llama
> ```

## 📊 Resources

The fine-tuning experiments were conducted using the following hardware configuration:

- **GPUs**: 8x NVIDIA A6000 (48GB VRAM each)
- **Training Time**: Approximately 8 hours / 4 hours in web navigation / code generation domain per model

> **Note**: The exact training time and memory requirements may vary depending on your specific model size and dataset configuration.