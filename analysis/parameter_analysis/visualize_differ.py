import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import os

# -------------------------- 1. 基础配置与初始化 --------------------------
DATA_PATH = "/home/xiaobei/jxk/agentic-rag-r1/Agentic-RAG-R1/output_eval/neuron/neuron_importance.pt"
SAVE_FIG_PATH = "/home/xiaobei/jxk/agentic-rag-r1/Agentic-RAG-R1/output_eval/neuron/heatmaps/"
os.makedirs(SAVE_FIG_PATH, exist_ok=True)

# 标签定义（action对应tags）
tag_keys = ["reasoning_tag", "search_tag", "summary_tag", "answer_tag", "backtrack_tag"]  # action
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

# 缓存数据 - 包含所有可对比元素
comparable_items = {}  # 键: 元素名称, 值: 数据矩阵(21,36)
global_min = None
global_max = None

# -------------------------- 2. 数据处理工具函数 --------------------------
def get_gathered_data(key_data, group_size=100):
    """获取单个标签的聚集数据（21组×36层）"""
    heatmap_data = []
    for layer_idx in range(36):
        layer_tensor = key_data[layer_idx].to(dtype=torch.float32)
        layer_np = layer_tensor.cpu().numpy()
        if len(layer_np) != 2048:
            raise ValueError(f"神经元数≠2048，实际为{len(layer_np)}")
        heatmap_data.append(layer_np)
    heatmap_data = np.array(heatmap_data).T  # (2048, 36)
    
    # 按每100个神经元分组聚集
    num_groups = (2048 + group_size - 1) // group_size  # 21组
    gathered_data = []
    for group_idx in range(num_groups):
        start = group_idx * group_size
        end = min((group_idx + 1) * group_size, 2048)
        group_mean = np.mean(heatmap_data[start:end, :], axis=0)
        gathered_data.append(group_mean)
    return np.array(gathered_data)  # (21, 36)

def init_data():
    """初始化数据：加载所有可对比元素（包括单个和总和）"""
    global comparable_items, global_min, global_max
    
    try:
        neuron_data = torch.load(DATA_PATH)
        print(f"📊 成功加载数据，包含 {len(neuron_data)} 个标签")
    except Exception as e:
        print(f"❌ 数据加载失败：{str(e)}")
        return False
    
    # 1. 加载单个元素数据（action和args）
    for key in all_target_keys:
        if key not in neuron_data:
            print(f"⚠️  数据中无 {key}，已跳过")
            continue
        key_data = neuron_data[key]
        if len(key_data) != 36:
            print(f"⚠️  {key} 层数≠36（实际{len(key_data)}），已跳过")
            continue
        try:
            gathered_data = get_gathered_data(key_data)
            comparable_items[key] = gathered_data
            print(f"🔍 已加载 {key}")
        except Exception as e:
            print(f"❌ {key} 处理失败：{str(e)}")
    
    # 2. 计算总和元素（sum_action和sum_args）
    valid_tags = [k for k in tag_keys if k in comparable_items]
    valid_args = [k for k in args_keys if k in comparable_items]
    
    if valid_tags:
        sum_action = np.sum([comparable_items[k] for k in valid_tags], axis=0)
        comparable_items["sum_action"] = sum_action
        print(f"📈 已计算 sum_action（{len(valid_tags)}个action的总和）")
    
    if valid_args:
        sum_args = np.sum([comparable_items[k] for k in valid_args], axis=0)
        comparable_items["sum_args"] = sum_args
        print(f"📈 已计算 sum_args（{len(valid_args)}个args的总和）")
    
    # 3. 确定全局极值（用于统一颜色范围）
    if comparable_items:
        all_values = []
        for data in comparable_items.values():
            all_values.extend([data.min(), data.max()])
        global_min = min(all_values)
        global_max = max(all_values)
    
    return len(comparable_items) > 0

# -------------------------- 3. 可视化工具函数 --------------------------
def plot_comparison(data1, name1, data2, name2):
    """对比两个元素：先显示各自热力图，再显示差异图"""
    # 1. 显示第一个元素的热力图
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    norm1 = TwoSlopeNorm(vcenter=0, vmin=global_min, vmax=global_max)
    im1 = ax1.imshow(data1, cmap=plt.cm.RdBu_r, norm=norm1, aspect='auto', interpolation='none')
    ax1.set_title(f'Heatmap: {name1}', fontweight='bold', pad=15)
    ax1.set_xlabel('Layer Index (0-35)', fontweight='bold')
    ax1.set_ylabel('Neuron Group (0-20)', fontweight='bold')
    ax1.set_xticks(np.arange(0, 36, 3))
    ax1.set_xticklabels(np.arange(0, 36, 3))
    ax1.set_yticks(np.arange(0, 21, 2))
    ax1.set_yticklabels([f"Group {i}" for i in range(0, 21, 2)])
    cbar1 = fig1.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label(f'Score (Range: {global_min:.2f} ~ {global_max:.2f})')
    plt.tight_layout()
    plt.savefig(f"{SAVE_FIG_PATH}/{name1}_heatmap.png", bbox_inches='tight')
    plt.show()
    
    # 2. 显示第二个元素的热力图
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    norm2 = TwoSlopeNorm(vcenter=0, vmin=global_min, vmax=global_max)
    im2 = ax2.imshow(data2, cmap=plt.cm.RdBu_r, norm=norm2, aspect='auto', interpolation='none')
    ax2.set_title(f'Heatmap: {name2}', fontweight='bold', pad=15)
    ax2.set_xlabel('Layer Index (0-35)', fontweight='bold')
    ax2.set_ylabel('Neuron Group (0-20)', fontweight='bold')
    ax2.set_xticks(np.arange(0, 36, 3))
    ax2.set_xticklabels(np.arange(0, 36, 3))
    ax2.set_yticks(np.arange(0, 21, 2))
    ax2.set_yticklabels([f"Group {i}" for i in range(0, 21, 2)])
    cbar2 = fig2.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label(f'Score (Range: {global_min:.2f} ~ {global_max:.2f})')
    plt.tight_layout()
    plt.savefig(f"{SAVE_FIG_PATH}/{name2}_heatmap.png", bbox_inches='tight')
    plt.show()
    
    # 3. 显示差异图（data1 - data2）
    diff = data1 - data2
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    norm3 = TwoSlopeNorm(vcenter=0, vmin=diff.min(), vmax=diff.max())
    im3 = ax3.imshow(diff, cmap=plt.cm.coolwarm, norm=norm3, aspect='auto', interpolation='none')
    ax3.set_title(f'Difference: {name1} - {name2}', fontweight='bold', pad=15)
    ax3.set_xlabel('Layer Index (0-35)', fontweight='bold')
    ax3.set_ylabel('Neuron Group (0-20)', fontweight='bold')
    ax3.set_xticks(np.arange(0, 36, 3))
    ax3.set_xticklabels(np.arange(0, 36, 3))
    ax3.set_yticks(np.arange(0, 21, 2))
    ax3.set_yticklabels([f"Group {i}" for i in range(0, 21, 2)])
    cbar3 = fig3.colorbar(im3, ax=ax3, shrink=0.8)
    cbar3.set_label(f'Difference (Range: {diff.min():.2f} ~ {diff.max():.2f})')
    plt.tight_layout()
    plt.savefig(f"{SAVE_FIG_PATH}/{name1}_vs_{name2}_diff.png", bbox_inches='tight')
    plt.show()

# -------------------------- 4. 交互选择功能 --------------------------
def print_comparable_list():
    """打印所有可对比的元素列表"""
    print("\n可对比的元素列表：")
    items = list(comparable_items.keys())
    for i, item in enumerate(items, 1):
        # 标记元素类型（action/args/总和）
        if item in tag_keys:
            print(f"{i}. {item} (action)")
        elif item in args_keys:
            print(f"{i}. {item} (args)")
        else:
            print(f"{i}. {item} (总和)")
    return items

def select_two_items():
    """让用户选择两个不同的元素进行对比"""
    items = print_comparable_list()
    if len(items) < 2:
        print("❌ 可对比元素不足2个，无法进行对比")
        return None, None
    
    # 选择第一个元素
    while True:
        try:
            choice1 = int(input("\n请选择第一个元素 (输入序号): ")) - 1
            if 0 <= choice1 < len(items):
                break
            else:
                print(f"❌ 请输入1-{len(items)}之间的数字")
        except ValueError:
            print("❌ 请输入有效的数字")
    
    # 选择第二个元素
    while True:
        try:
            choice2 = int(input("请选择第二个元素 (输入序号): ")) - 1
            if 0 <= choice2 < len(items):
                if choice1 != choice2:
                    break
                else:
                    print("❌ 两个元素不能相同，请重新选择")
            else:
                print(f"❌ 请输入1-{len(items)}之间的数字")
        except ValueError:
            print("❌ 请输入有效的数字")
    
    return items[choice1], items[choice2]

def main_menu():
    """主菜单"""
    while True:
        print("\n" + "="*60)
        print("           神经元重要性对比工具 (支持所有元素两两对比)")
        print("="*60)
        print("1. 查看所有可对比元素列表")
        print("2. 选择两个元素进行对比（支持任意组合：action/args/总和）")
        print("3. 退出程序")
        print("="*60)
        
        choice = input("请输入选项 (1-3): ")
        
        if choice == '1':
            print_comparable_list()
        
        elif choice == '2':
            name1, name2 = select_two_items()
            if name1 and name2:
                print(f"\n正在对比：{name1} 和 {name2} ...")
                plot_comparison(
                    data1=comparable_items[name1],
                    name1=name1,
                    data2=comparable_items[name2],
                    name2=name2
                )
        
        elif choice == '3':
            print("👋 程序已退出")
            break
        
        else:
            print("❌ 无效的选项，请重新输入")

# -------------------------- 5. 程序入口 --------------------------
if __name__ == "__main__":
    print("🚀 增强版神经元重要性对比工具启动中...")
    if init_data():
        print(f"\n✅ 数据初始化完成，共加载 {len(comparable_items)} 个可对比元素")
        main_menu()
    else:
        print("❌ 数据初始化失败，程序退出")
    