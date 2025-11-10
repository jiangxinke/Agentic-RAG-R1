import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import os

# -------------------------- 1. 基础配置与初始化 --------------------------
DATA_PATH = "/home/xiaobei/qrh/Agentic-RAG-R1/output_eval/neuron/neuron_importance.pt"
SAVE_FIG_PATH = "/home/xiaobei/jxk/agentic-rag-r1/Agentic-RAG-R1/output_eval/neuron/heatmaps/"
os.makedirs(SAVE_FIG_PATH, exist_ok=True)

tag_keys = ["reasoning_tag", "search_tag", "summary_tag", "answer_tag", "backtrack_tag"]
args_keys = ["reasoning_arg", "search_arg", "summary_arg", "answer_arg", "backtrack_arg"]
all_target_keys = tag_keys + args_keys

plt.rcParams.update({
    'font.sans-serif': ['DejaVu Sans'],
    'axes.linewidth': 0.8,
    'font.size': 10,
    'figure.dpi': 300,
    'figure.facecolor': 'white'
})

comparable_items = {}
global_min = None
global_max = None

# -------------------------- 2. 数据处理工具函数 --------------------------
def get_gathered_data(key_data, group_size=50):
    heatmap_data = []
    for layer_idx in range(36):
        layer_tensor = key_data[layer_idx].to(dtype=torch.float32)
        layer_np = layer_tensor.cpu().numpy()
        if len(layer_np) != 2048:
            raise ValueError(f"神经元数≠2048，实际为{len(layer_np)}")
        heatmap_data.append(layer_np)
    heatmap_data = np.array(heatmap_data).T  # (2048, 36)

    # binarize
    heatmap_data = np.where(heatmap_data>=0, np.ones_like(heatmap_data), np.zeros_like(heatmap_data))

    num_groups = (2048 + group_size - 1) // group_size
    gathered_data = []
    for group_idx in range(num_groups):
        start = group_idx * group_size
        end = min((group_idx + 1) * group_size, 2048)
        group_mean = np.mean(heatmap_data[start:end, :], axis=0)
        gathered_data.append(group_mean)
    return np.array(gathered_data)  # (21, 36)

def init_data():
    global comparable_items, global_min, global_max
    
    try:
        neuron_data = torch.load(DATA_PATH)
        print(f"📊 成功加载数据，包含 {len(neuron_data)} 个标签")
    except Exception as e:
        print(f"❌ 数据加载失败：{str(e)}")
        return False
    
    for key in all_target_keys:
        if key not in neuron_data:
            continue
        key_data = neuron_data[key]
        if len(key_data)!=36:
            continue
        gathered_data = get_gathered_data(key_data)
        comparable_items[key] = gathered_data

    valid_tags = [k for k in tag_keys if k in comparable_items]
    valid_args = [k for k in args_keys if k in comparable_items]

    if valid_tags:
        comparable_items["sum_action"] = np.mean([comparable_items[k] for k in valid_tags], axis=0)
    if valid_args:
        comparable_items["sum_args"] = np.mean([comparable_items[k] for k in valid_args], axis=0)

    all_values = []
    for data in comparable_items.values():
        all_values.extend([data.min(), data.max()])
    global_min = min(all_values)
    global_max = max(all_values)

    return True

# -------------------------- 3. 可视化工具函数 --------------------------
def plot_comparison(data1, name1, data2, name2):
    # norm 用 Normalize 因为 now 全部是 0~1 binary aggregation
    norm_main = Normalize(vmin=0, vmax=1)

    fig1, ax1 = plt.subplots(figsize=(12, 8))
    im1 = ax1.imshow(data1, cmap=plt.cm.RdBu_r, norm=norm_main, aspect='auto', interpolation='none')
    ax1.set_title(f'Heatmap: {name1}', fontweight='bold', pad=15)
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    plt.tight_layout()
    plt.savefig(f"{SAVE_FIG_PATH}/{name1}_heatmap.png", bbox_inches='tight')
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(12, 8))
    im2 = ax2.imshow(data2, cmap=plt.cm.RdBu_r, norm=norm_main, aspect='auto', interpolation='none')
    ax2.set_title(f'Heatmap: {name2}', fontweight='bold', pad=15)
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    plt.tight_layout()
    plt.savefig(f"{SAVE_FIG_PATH}/{name2}_heatmap.png", bbox_inches='tight')
    plt.show()

    diff = data1 - data2
    norm_diff = TwoSlopeNorm(vcenter=0, vmin=diff.min(), vmax=diff.max())
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    im3 = ax3.imshow(diff, cmap=plt.cm.coolwarm, norm=norm_diff, aspect='auto', interpolation='none')
    ax3.set_title(f'Difference: {name1} - {name2}', fontweight='bold', pad=15)
    plt.colorbar(im3, ax=ax3, shrink=0.8)
    plt.tight_layout()
    plt.savefig(f"{SAVE_FIG_PATH}/{name1}_vs_{name2}_diff.png", bbox_inches='tight')
    plt.show()

# -------------------------- 4. 菜单 --------------------------
def print_comparable_list():
    print("\n可对比的元素列表：")
    items = list(comparable_items.keys())
    for i,item in enumerate(items,1):
        print(f"{i}. {item}")
    return items

def select_two_items():
    items = print_comparable_list()
    c1 = int(input("\n请选择第一个 (序号): "))-1
    c2 = int(input("请选择第二个 (序号): "))-1
    return items[c1], items[c2]

def main_menu():
    while True:
        print("\n1. 查看列表")
        print("2. 对比两个")
        print("3. exit")
        choice=input("选择: ")

        if choice=='1':
            print_comparable_list()
        elif choice=='2':
            name1,name2 = select_two_items()
            plot_comparison(comparable_items[name1], name1, comparable_items[name2], name2)
        else:
            break

if __name__ == "__main__":
    print("🚀 启动中...")
    if init_data():
        print("✅ 初始化完成")
        main_menu()
