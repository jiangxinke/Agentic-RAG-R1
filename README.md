# In gjr branch

# TODO

- Support multi-gpu--grpo_trainer_mu_GPU的loss.backward总是出现段错误，检查多卡错误
- 从main函数中分割出args参数代码出来
- 检查generation_interrupt_new的代码，处理过程是否有错误
- 检查evaluation代码


<!-- - Load Xiaobei CKP -->
<!-- - 注意balance -->

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