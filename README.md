# Unintended Misalignment from Agentic Fine-Tuning

![main_figure](./asset/main_figure.png)

## Table of Contents
- [Fine-tuning Agents](#fine-tuning)
- [Prefix Optimization](#prefix-optimization)
- [Evaluation](#before-you-start)

## Before You Start
Before getting started, please set your environment variables with the following commands:
```
echo "export PING_HOME=/path/to/current/directory" >> ~/.bashrc
source ~/.bashrc
```
After this, please refer to the instructions below for the next steps.

## Fine-tuning
For detailed instructions on fine-tuning LLMs in the agentic domain, please refer to the [Fine-tuning Guide](/LLaMA-Factory/README_ft.md).

## Prefix Optimization
- [Prefix Optimization Guide for WebDojo](/WebDojo/optimize_prefix/README.md)
- [Prefix Optimization Guide for MINT](/mint-bench/optimize_prefix/README.md)
## Evaluation

Each benchmark requires its own setup and configuration.
Please refer to the manual for each benchmark to evaluate the agent.
- [WebArena-Lite](/VisualAgentBench/VAB-WebArena-Lite/README.md)
- [WebDojo](/WebDojo/README.md)
- [RedCode-Exec](/RedCode/README.md)
- [MINT-ALFWorld](/mint-bench/README.md)

## Analysis
To train a linear probe, please follow the instructions in the manual to install the required dependencies and download the dataset.
- [Linear Probe Training](/probe_training/README.md)
