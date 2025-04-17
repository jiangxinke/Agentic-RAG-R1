# Agentic RAG-R1: Enhance Agentic RAG Reasoning Capacity via Reinforcement Learning

## Table of Contents
- [Introduction](#introduction)
- [What is Agentic RAG?](#what-is-agentic-rag)
- [Architecture](#architecture)
- [Training Strategy](#training-strategy)
- [Features](#features)
- [Installation](#installation)
- [Folder Structure](#folder-structure)
- [Quick Start](#quick-start)
- [Results](#results)
- [Roadmap](#roadmap)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)
- [License](#license)

## Introduction

Agentic RAG‑R1 is an open‑source initiative to build an Agentic Retrieval‑Augmented Generation (RAG) system by endowing a base language model with autonomous search & reasoning skills through reinforcement learning (currently using the GRPO algorithm).

**Chinese Language Versions:**
![Chinese version results](https://github.com/user-attachments/assets/a6e42d35-4fec-43b9-9a04-3d102e544e20)

**English Language Versions:**
![English version results](https://github.com/user-attachments/assets/40f11648-bf46-4cd3-873c-78ca63069499)

## What is Agentic RAG?

Agentic RAG combines two powerful concepts:

- **Retrieval‑Augmented Generation (RAG)**: Combines generative power with on‑the‑fly retrieval from external knowledge bases, ensuring factual and up‑to‑date answers.
- **Agentic AI**: Gives the model the ability to decide when to retrieve, what to retrieve, and how to weave the retrieved evidence into its reasoning.

![Agentic RAG concept](https://github.com/user-attachments/assets/7b4b6559-b395-4de0-8326-ad0fca2e671a)

## Architecture

Our architecture is inspired by TC‑RAG and features an agent memory stack that orchestrates the full deliberation loop, supporting the following actions:

1. Plan (❌)
2. Reasoning (✅)
3. Backtrack (✅)
4. Summary (✅)
5. Tool Observation – wiki/document/knowledge‑graph search, etc. (✅)
6. Conclusion (✅)

![Architecture diagram](https://github.com/user-attachments/assets/53dfae56-6c59-488f-9313-7688d5839077)

## Training Strategy

Motivated by DeepSeek-R1, we apply GRPO (Generalized Relevance Policy Optimization) to reinforce the agent's choice of reasoning steps and retrieval actions, effectively boosting both search depth and answer quality.

![Training strategy diagram](https://github.com/user-attachments/assets/9880394a-f16a-4acd-84c8-db9f4f7d8433)

### Rollout Generation

![Rollout generation diagram](https://github.com/user-attachments/assets/21d90097-f7a4-46ef-a442-c8a0a778bab4)

## Features

- **LoRA Tuning Support**: Fine-tune efficiently with Low-Rank Adaptation
- **Custom Agent Tools**: Integrate your own tools and personal RAG datasets
- **Distributed Training**: Support for Deepspeed Zero 2 Stage and Zero 3 Stage
- **Efficient Resource Usage**: Support for models up to 32B parameters using only 2 A100 GPUs
- **Tool Calling Reward**: Enhanced reward model that includes:
  - Accuracy reward
  - Format reward
  - RAG accuracy reward using the RAGAS framework

The total reward is calculated as:

$$r_{total} = r_{accuracy} + r_{format} + r_{rag}$$

- **TCRAG Integration**: Uses TCRAG as the rollout generator

## Installation

We use conda to manage the environment. Follow these steps to set up:

```bash
conda create -n AgenticRAG python=3.11 -y
conda activate AgenticRAG 
pip install -r requirements.txt
```

### Tools Environment (Optional)

We provide our search tool repository [ArtSearch](https://github.com/Artessay/ArtSearch) as the search engine, which supports retrieval of information from Wikipedia. You can follow the instructions in that repository to deploy a local instance of the search system.

## Folder Structure

```
.
├── ArtSearch                 # Search tool integration
├── checkpoints               # Model checkpoints
├── examples                  # Example use cases
├── experiments
│   ├── evaluation            # Evaluation scripts and results
│   └── training              # Training configurations
├── README.md
├── requirements.txt
├── script
│   ├── evaluation            # Evaluation scripts
│   ├── run_server.sh         # Server deployment script
│   └── training              # Training scripts
├── service
│   ├── chat_client.py        # Client for interacting with the model
│   └── chat_server.py        # Server for hosting the model
├── src
│   ├── config                # Configuration files
│   ├── data                  # Data processing utilities
│   ├── evaluation            # Evaluation metrics and tools
│   ├── models                # Model definitions
│   ├── train.py              # Main training script
│   └── utils                 # Utility functions
```

## Quick Start

### Training

[Training instructions to be added]

### Inference

[Inference instructions to be added]

## Results

### Experiment Log on Qwen 2.5-7B-Instruct

![Experiment log](https://github.com/user-attachments/assets/591d61aa-d4e4-45e9-a48a-77a01858a24b)

We have made our training logs publicly available at: [SwanLab Training Log](https://swanlab.cn/@devilran/xiaobeir1/runs/ipuoxctxo764rvub20d6h/chart)

### Results on MedQA Test Set

Our Qwen 2.5-7B-Instruct model was evaluated on the MedQA test set using Qwen‑2.5‑72B as the judge:

[Results table to be added]

## Roadmap

- [ ] Support model quantification
- [ ] Add more tools
- [ ] [Additional planned features]

## Acknowledgements

The concept of Agentic-RAG-R1 is inspired by Deepseek-R1 and TC-RAG. We sincerely appreciate the efforts of these teams for their contributions to open-source research and development.

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{Agentic_RAG_R1,
  title       = {Agentic RAG-R1: Enhance Agentic RAG Reasoning Capacity via Reinforcement Learning},
  author      = {Xinke Jiang, Jiaran Gao, Rihong Qiu, Wentao Zhang, Yue Fang},
  year        = {2025},
  howpublished= {\url{https://github.com/jiangxinke/Agentic-RAG-R1}},
  note        = {GitHub repository},
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
