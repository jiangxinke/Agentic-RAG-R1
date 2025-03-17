# 环境配置

conda create --name xiaobeir1 \
    python=3.11 \
    pytorch-cuda=12.1 \
    pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
    -y
conda activate xiaobeir1

pip install unsloth


# 运行方式：
```
python GRPO.py
```