# In gjr branch

# TODO

- Support multi-gpu--grpo_trainer_mu_GPU的loss.backward总是出现段错误，检查多卡错误
- 从main函数中分割出args参数代码出来
- 检查generation_interrupt_new的代码，处理过程是否有错误
- 检查evaluation代码


- 将reference model不报错embedding 2D
- 将reference model实现zero 3优化

~~any~~
modelsaver 间隔 10 个 iter 保存一次
demo

swandb
eval-post

<!-- - Load Xiaobei CKP -->
<!-- - 注意balance -->


done 3. 上传 github 新分支 demo acc demo  README
done 1. 微调前的多卡推断 以及带 id 带正确错误的 100
2. 统计微调前和微调后的长度
4. 统计 reward swandb 重新训练 100step

======================================================

20250405
1. 写一个能现成跑 demo 的脚本
2. 更新规模 训练 2000 条  测试 1000 条
3. 微调前；微调前+search；微调后+search
7. 观察 backtrace 有没有什么效果
done 4. 统计长度、答案格式准确率、search 调用次数、reasoning、backtrack。loss reward
done 5. swandb
done 6. 配置 dot 文件
done 8.TypeError: Tools.Wiki_RAG() got an unexpected keyword argument 'query' try
done 9.2222 Web_RAG None
done 10.if sum_steps % 2 == 0 and sum_steps > current_step: 参数
~~~
problem: Swandb 只记录了第一张卡的
======================================================



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