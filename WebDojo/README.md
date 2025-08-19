# WebDojo: A Benchmark for Evaluating the Safety of Web Navigation Agents

This repository contains test websites designed to evaluate the safety and ethical behavior of LLM-based web control agents. The purpose is to assess how these agents handle potentially harmful or illegal content requests, and whether they maintain appropriate ethical boundaries during web interactions.

## Table of Contents

- [Installation guideline](#installation-guideline)
- [Web deployment](#web-deployment)
- [Evaluation](#evaluation)
- [Acknowledgements](#acknowledgements)


## Installation Guideline

```bash
# set path
echo "export WEBDOJO_HOME=/path/to/current/directory" >> ~/.bashrc
source ~/.bashrc

# ake conda env
conda create -n webdojo python=3.10
conda activate webdojo

# install packages
pip install -r requirements.txt

# install browsergym
cd browsergym
make install
cd ..

# install playwright
playwright install chromium

# install webdojo
pip install -e .
```

## Web Deployment

To deploy the websites locally, please enter the following command:
```bash
# Start local server from repository root
python3 -m http.server 8000
```
Afterwards, you can access the websites at http://localhost:8000.

Using the port you selected, run the following command:
```bash
# Start local server from repository root
python3 web_safety/deploy.py {port}
# Example 
python3 web_safety/deploy.py 8000
```

## Evaluation

Evaluate the safety of the LLM agent using the command below:
```bash
python3 web_safety/main.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --model_type base \
  --category both \
  --save_results True \
  --split test \
  --system_prompt default \
  --prefix "" \
  --port 8000
```
For a description of each parameter and available values, please refer to main/deploy.py.

## Guard

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

## Warning

This repository contains simulated illegal content for testing purposes only. It should not be used for any real-world transactions or activities. 

## Web Deployment
Some codes are referred from the related work below:
- [BrowserGym](https://github.com/ServiceNow/BrowserGym)
- [WebRL](https://github.com/THUDM/WebRL)