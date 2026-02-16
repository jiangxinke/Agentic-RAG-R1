# AgenticRAG-R1

## 🛠️ Environment Setup

Follow the steps below to set up the development environment.

### 1. Create Conda Environment
```bash
conda create -n verl-sglang python=3.12 -y
conda activate verl-sglang
```

### 2. Install PyTorch
Install PyTorch and related components using the specific Aliyun mirror sources:
```bash
python -m pip install -U pip wheel setuptools

python -m pip install \
"https://mirrors.aliyun.com/pytorch-wheels/cu129/torch-2.9.1%2Bcu129-cp312-cp312-manylinux_2_28_x86_64.whl" \
"https://mirrors.aliyun.com/pytorch-wheels/cu129/torchvision-0.24.1%2Bcu129-cp312-cp312-manylinux_2_28_x86_64.whl" \
"https://mirrors.aliyun.com/pytorch-wheels/cu129/torchaudio-2.9.1%2Bcu129-cp312-cp312-manylinux_2_28_x86_64.whl"
```

### 3. Install Flash Attention
Download the appropriate wheel version from [flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels/releases) and install it:

```bash
# Example command (ensure the filename matches your download)
python -m pip install flash_attn-2.7.4+cu128torch2.9-cp312-cp312-linux_x86_64.whl
```

### 4. Install SGLang
```bash
python -m pip install --pre sglang
python -m pip install --no-deps "sglang[openai,srt]==0.5.5"   
python -m pip install "sglang[openai,srt]==0.5.6.post2"
```


### 5. Install vLLM
> **⚠️ Important:** Install vLLM with `--no-deps`. Do **NOT** add dependencies here, otherwise it may alter the Torch version and cause conflicts.

```bash
python -m pip install --no-deps vllm==0.12.0
```

```bash
python -m pip install ray
python -m pip install tensordict
python -m pip install omegaconf
python -m pip install hydra-core
python -m pip install torchdata
python -m pip install codetiming
python -m pip install peft
python -m pip install cachetools
python -m pip install cbor2
python -m pip install swanlab
```

---

## 🚀 Training

### 1. Configuration
Before running the training script, update the environment variables and paths in `./verl/spr1_scripts/local_config.sh`.

**Key configurations to check:**
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export RAY_TMPDIR=""
export SWANLAB_API_KEY=""

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"
TOOL_CONFIG="$CONFIG_PATH/tool_config/search_tool_config.yaml"

PROJECT_NAME=''
EXPERIMENT_NAME=''

MODEL_PATH=""
TRAIN_DATA_PATH=""
TEST_DATA_PATH=""
```

### 2. Run Training
Navigate to the `verl` directory and execute the script:

```bash
cd verl
./spr1_scripts/run_qwen2.5-3b_instruct_search_multiturn.sh
```

---

## 📚 Acknowledgement

*   [VeRL: Volcano Engine Reinforcement Learning for LLMs](https://github.com/volcengine/verl)
*   [ArtSearch: A Local Search System for Wikipedia](https://github.com/Artessay/ArtSearch)