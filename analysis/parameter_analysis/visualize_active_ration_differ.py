import json
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

# ---------------------- 1. 路径与参数配置 ----------------------
tag_keys = ["reasoning_tag", "search_tag", "summary_tag", "answer_tag", "backtrack_tag"]  # action相关
args_keys = ["reasoning_arg", "search_arg", "summary_arg", "answer_arg", "backtrack_arg"]
data_path = "/home/xiaobei/qrh/Agentic-RAG-R1/output_eval/neuron/neuron_active_ratio.json"
save_dir = "/home/xiaobei/jxk/agentic-rag-r1/Agentic-RAG-R1/output_eval/neuron/layer_importances_plots/"
os.makedirs(save_dir, exist_ok=True)

# 计算top 20%的层数（32层的20%是6-7层）
top_percent = 0.2
num_top_layers = int(32 * top_percent)
if num_top_layers == 0:
    num_top_layers = 1  # 确保至少有1个顶层

# 标准化配置
scaler = MinMaxScaler(feature_range=(-1, 1))


# ---------------------- 2. 数据读取与处理 ----------------------
with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

layers = np.arange(32)

# 原始数据存储
tag_data = {key: np.array([data[key][str(i)] for i in range(32)]) for key in tag_keys}
args_data = {key: np.array([data[key][str(i)] for i in range(32)]) for key in args_keys}

# 计算每个layer的action总和和args总和
action_sums = np.sum([tag_data[key] for key in tag_keys], axis=0)
args_sums = np.sum([args_data[key] for key in args_keys], axis=0)

# ---------------------- 修复 1：标准化后再做差 ----------------------
action_sums_scaled = scaler.fit_transform(action_sums.reshape(-1, 1)).flatten()
args_sums_scaled = scaler.fit_transform(args_sums.reshape(-1, 1)).flatten()

# 差异 = 归一化 action - 归一化 args
differ = action_sums_scaled - args_sums_scaled

# 差异无需再归一化
differ_scaled = differ.copy()

# ---------------------- top 层选择 ----------------------
top_action_layers = np.argsort(action_sums)[-num_top_layers:]
top_args_layers = np.argsort(args_sums)[-num_top_layers:]
top_differ_layers = np.argsort(np.abs(differ))[-num_top_layers:]  # 差异最大的层


# ---------------------- 3. 绘图配置 ----------------------
plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans"],
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.figsize": (16, 18),
    "axes.spines.top": False,
    "axes.spines.right": False
})
colors = {
    "action": "#1f77b4",
    "args": "#ff7f0e",
    "differ": "#d62728",
    "top": "#9467bd"
}
bar_width = 0.6


# ---------------------- 4. 绘制action和args总和对比图 ----------------------
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(16, 18))

# Action总和
ax1.set_title("Sum of Action Layer Importance (Scaled)", fontsize=14, fontweight="bold")
for i, layer in enumerate(layers):
    color = colors["top"] if layer in top_action_layers else colors["action"]
    ax1.bar(layer, action_sums_scaled[i], color=color, alpha=0.8,
            edgecolor="black", linewidth=0.7, width=bar_width)
ax1.set_ylim(-1.2, 1.2)
ax1.set_yticks([-1, -0.5, 0, 0.5, 1])
ax1.set_ylabel("Scaled Importance", fontsize=12)
ax1.grid(axis="y", alpha=0.3, linestyle="--")
ax1.axhline(y=0, color="black", linewidth=0.5, alpha=0.5)
ax1.text(0.02, 0.95, f"Top {int(top_percent*100)}% layers highlighted",
         transform=ax1.transAxes, fontsize=10,
         bbox=dict(facecolor='white', alpha=0.8))

# Args总和
ax2.set_title("Sum of Args Layer Importance (Scaled)", fontsize=14, fontweight="bold")
for i, layer in enumerate(layers):
    color = colors["top"] if layer in top_args_layers else colors["args"]
    ax2.bar(layer, args_sums_scaled[i], color=color, alpha=0.8,
            edgecolor="black", linewidth=0.7, width=bar_width)
ax2.set_ylim(-1.2, 1.2)
ax2.set_yticks([-1, -0.5, 0, 0.5, 1])
ax2.set_ylabel("Scaled Importance", fontsize=12)
ax2.grid(axis="y", alpha=0.3, linestyle="--")
ax2.axhline(y=0, color="black", linewidth=0.5, alpha=0.5)

# 差异图
ax3.set_title("Difference Between Action and Args (Scaled Action - Scaled Args)",
              fontsize=14, fontweight="bold")
for i, layer in enumerate(layers):
    ax3.bar(layer, differ_scaled[i], color=colors["differ"], alpha=0.8,
            edgecolor="black", linewidth=0.7, width=bar_width)
ax3.set_ylim(-1.2, 1.2)
ax3.set_yticks([-1, -0.5, 0, 0.5, 1])
ax3.set_xlabel("Layer Index (0-31)", fontsize=12)
ax3.set_ylabel("Difference", fontsize=12)
ax3.grid(axis="y", alpha=0.3, linestyle="--")
ax3.axhline(y=0, color="black", linewidth=0.5, alpha=0.5)
ax3.set_xticks(layers)
ax3.set_xticklabels(layers, rotation=0)

# 保存图
plt.tight_layout()
sum_diff_save_path = os.path.join(save_dir, "action_args_sum_and_difference.png")
plt.savefig(sum_diff_save_path, dpi=300, bbox_inches="tight")
plt.close()


# ---------------------- 5. 绘制 top 层对比表格 ----------------------
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

table_data = [
    ["Top Layer Type", "Layer Indices", "Average Value"]
]

top_action_avg = np.mean(action_sums[top_action_layers])
table_data.append(["Action", ", ".join(map(str, sorted(top_action_layers))), f"{top_action_avg:.2f}"])

top_args_avg = np.mean(args_sums[top_args_layers])
table_data.append(["Args", ", ".join(map(str, sorted(top_args_layers))), f"{top_args_avg:.2f}"])

top_differ_avg = np.mean(np.abs(differ[top_differ_layers]))
table_data.append(["Difference", ", ".join(map(str, sorted(top_differ_layers))), f"{top_differ_avg:.2f}"])

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                 colWidths=[0.3, 0.5, 0.2], edges='closed')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.5)

plt.title(f"Top {int(top_percent*100)}% Layer Comparison", fontsize=14, fontweight="bold", y=1.1)

# ---------------------- 修复 2：文件名加入 top 百分比 ----------------------
table_save_path = os.path.join(save_dir, f"top_layers_comparison_top{int(top_percent*100)}.png")
plt.savefig(table_save_path, dpi=300, bbox_inches="tight")
plt.close()


# ---------------------- 6. 结果提示 ----------------------
print("高级可视化完成！图片保存路径：")
print(f"1. 总和与差异对比图：{sum_diff_save_path}")
print(f"2. Top {int(top_percent*100)}% 层对比表：{table_save_path}")
print(f"\nTop {int(top_percent*100)}% 重要层数量：{num_top_layers}")
print(f"Action 重要层：{sorted(top_action_layers)}")
print(f"Args 重要层：{sorted(top_args_layers)}")
print(f"差异最大层：{sorted(top_differ_layers)}")
