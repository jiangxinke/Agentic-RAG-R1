# README

## environment

step01: create new conda environment
```
conda create -n verl-sglang python=3.12 -y
conda activate verl-sglang
```

step02: install torch
```
python -m pip install -U pip wheel setuptools

python -m pip install \
  "https://mirrors.aliyun.com/pytorch-wheels/cu129/torch-2.9.1%2Bcu129-cp312-cp312-manylinux_2_28_x86_64.whl" \
  "https://mirrors.aliyun.com/pytorch-wheels/cu129/torchvision-0.24.1%2Bcu129-cp312-cp312-manylinux_2_28_x86_64.whl" \
  "https://mirrors.aliyun.com/pytorch-wheels/cu129/torchaudio-2.9.1%2Bcu129-cp312-cp312-manylinux_2_28_x86_64.whl"
```

step03: install flash-attn

download proper version wheel from [flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels/releases)

```
python -m pip install flash_attn-2.7.4+cu128torch2.9-cp312-cp312-linux_x86_64.whl
```

step04: install sglang
```
python -m pip install --pre sglang
```

step05: install vllm with –no-deps
```
python -m pip install --no-deps vllm==0.12.0
```
Note that dependencies must not be added here; otherwise, it will cause changes in the torch version

## wiki

参考：[ArtSearch](https://github.com/Artessay/ArtSearch)
