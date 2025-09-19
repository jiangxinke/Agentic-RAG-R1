import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import os

# -------------------------- 1. 基础配置 --------------------------
DATA_PATH = "/home/xiaobei/jxk/agentic-rag-r1/Agentic-RAG-R1/output_eval/neuron/neuron_importance.pt"
SAVE_FIG_PATH = "/home/xiaobei/jxk/agentic-rag-r1/Agentic-RAG-R1/output_eval/neuron/heatmaps/"
os.makedirs(SAVE_FIG_PATH, exist_ok=True)

# 所有需要可视化的标签
tag_keys = ["reasoning_tag", "search_tag", "summary_tag", "answer_tag", "backtrack_tag"]
args_keys = ["reasoning_arg", "search_arg", "summary_arg", "answer_arg", "backtrack_arg"]
all_target_keys = tag_keys + args_keys

# 全局绘图配置
plt.rcParams.update({
    'font.sans-serif': ['DejaVu Sans'],
    'axes.linewidth': 0.8,
    'font.size': 10,
    'figure.dpi': 300,
    'figure.facecolor': 'white'
})

# -------------------------- 2. 核心函数：按每100个神经元聚集并绘图 --------------------------
def plot_gathered_heatmap(key_name, key_data, group_size=100):
    """
    按每100个神经元分组聚集（计算平均值），生成热力图
    x轴=Layer（0-35），y轴=神经元组（0-20，每组含100个神经元）
    """
    # 1. 构建原始36×2048数据矩阵 → 转置为2048×36
    heatmap_data = []
    for layer_idx in range(36):
        layer_tensor = key_data[layer_idx].to(dtype=torch.float32)
        layer_np = layer_tensor.cpu().numpy()
        if len(layer_np) != 2048:
            raise ValueError(f"{key_name} 第{layer_idx}层神经元数≠2048，实际为{len(layer_np)}")
        heatmap_data.append(layer_np)
    heatmap_data = np.array(heatmap_data).T  # 形状：(2048, 36)

    # 2. 按每100个神经元分组，计算每组平均值（核心聚集步骤）
    num_neurons = 2048
    num_groups = (num_neurons + group_size - 1) // group_size  # 向上取整：2048→21组
    gathered_data = []
    
    for group_idx in range(num_groups):
        # 计算当前组的神经元索引范围
        start = group_idx * group_size
        end = min((group_idx + 1) * group_size, num_neurons)  # 最后一组可能不足100个
        group_neurons = heatmap_data[start:end, :]  # 提取当前组的所有神经元（形状：(n, 36)，n≤100）
        
        # 计算组内平均值（沿神经元维度聚合）
        group_mean = np.sum(group_neurons, axis=0)  # 形状：(36,)
        gathered_data.append(group_mean)
    
    gathered_data = np.array(gathered_data)  # 最终形状：(21, 36)

    # 3. 创建画布（适配21组×36层）
    fig, ax = plt.subplots(figsize=(12, 8))  # 宽12（层），高8（神经元组）

    # 4. 固定颜色范围-10~10
    norm = TwoSlopeNorm(vmin=-10, vmax=10, vcenter=0)
    cmap = plt.cm.RdBu_r

    # 5. 绘制聚集后的热力图
    im = ax.imshow(gathered_data, cmap=cmap, norm=norm, aspect='auto', interpolation='none')

    # 6. 坐标轴标签（明确分组信息）
    ax.set_xlabel('Layer Index (0-35)', fontweight='bold', labelpad=10)
    ax.set_ylabel(f'Neuron Group (0-20, each group={group_size} neurons)', fontweight='bold', labelpad=10)
    ax.set_title(
        f'Gathered Neuron Importance (per {group_size} neurons) - {key_name}\n(21 Groups × 36 Layers, Range: -10~10)',
        fontweight='bold', fontsize=12, pad=15
    )

    # 7. 刻度调整（适配分组后的维度）
    # x轴（层）：每3层标1个刻度
    ax.set_xticks(np.arange(0, 36, 3))
    ax.set_xticklabels(np.arange(0, 36, 3))
    # y轴（神经元组）：每2组标1个刻度，并标注对应神经元范围
    ax.set_yticks(np.arange(0, num_groups, 2))
    ax.set_yticklabels([
        f"Group {i} (={i*group_size}-{(i+1)*group_size-1})" 
        for i in range(0, num_groups, 2)
    ])

    # 8. 添加颜色条
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(f'Avg Importance Score (per {group_size} neurons, Range: -10~10)', fontweight='bold', labelpad=10)

    # 9. 保存图片
    save_path = f"{SAVE_FIG_PATH}/{key_name}_gathered_{group_size}_heatmap.png"
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ 已保存聚集热力图：{save_path}")

# -------------------------- 3. 主程序 --------------------------
if __name__ == "__main__":
    try:
        neuron_data = torch.load(DATA_PATH)
        print(f"📊 成功加载数据，包含 {len(neuron_data)} 个标签")
    except Exception as e:
        print(f"❌ 数据加载失败：{str(e)}")
        exit()

    for key in all_target_keys:
        print(f"\n正在处理标签：{key}")
        if key not in neuron_data:
            print(f"⚠️  数据中无 {key}，已跳过")
            continue
        key_data = neuron_data[key]
        if len(key_data) != 36:
            print(f"⚠️  {key} 层数≠36（实际{len(key_data)}），已跳过")
            continue
        try:
            plot_gathered_heatmap(key, key_data, group_size=100)  # 每100个神经元聚集
        except Exception as e:
            print(f"❌ {key} 处理失败：{str(e)}")

    print("\n🎉 所有聚集热力图处理完成！保存路径：", SAVE_FIG_PATH)
    