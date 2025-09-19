import json
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler  # 用于数据标准化

# ---------------------- 1. 路径与参数配置 ----------------------
tag_keys = ["reasoning_tag", "search_tag", "summary_tag", "answer_tag", "backtrack_tag"]
args_keys = ["reasoning_arg", "search_arg", "summary_arg", "answer_arg", "backtrack_arg"]
# 更新数据路径为新提供的路径
data_path = "/home/xiaobei/qrh/Agentic-RAG-R1/output_eval/neuron/neuron_active_ratio.json"
save_dir = "/home/xiaobei/jxk/agentic-rag-r1/Agentic-RAG-R1/output_eval/neuron/layer_importances_plots"
os.makedirs(save_dir, exist_ok=True)  # 确保保存目录存在

# 标准化配置：将每组数据缩放到 [-1, 1] 范围（放大差异，保留正负）
scaler = MinMaxScaler(feature_range=(-1, 1))


# ---------------------- 2. 数据读取与标准化处理 ----------------------
with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 提取前32层原始数据 + 标准化数据（每组单独缩放，最大化组内差异）
layers = np.arange(32)
tag_data = {"raw": {}, "scaled": {}}  # raw=原始值，scaled=标准化后的值
args_data = {"raw": {}, "scaled": {}}

# Tag组数据处理
for key in tag_keys:
    raw = np.array([data[key][str(i)] for i in range(32)]).reshape(-1, 1)  # 转为2D数组（适配scaler）
    scaled = scaler.fit_transform(raw).flatten()  # 标准化后展平为1D
    tag_data["raw"][key] = raw.flatten()
    tag_data["scaled"][key] = scaled

# Args组数据处理
for key in args_keys:
    raw = np.array([data[key][str(i)] for i in range(32)]).reshape(-1, 1)
    scaled = scaler.fit_transform(raw).flatten()
    args_data["raw"][key] = raw.flatten()
    args_data["scaled"][key] = scaled


# ---------------------- 3. 绘图配置（突出区分度） ----------------------
plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans"],
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.figsize": (8, 22),
    "axes.spines.top": False,
    "axes.spines.right": False
})
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
bar_width = 0.9  # 柱子宽度，增强视觉效果


# ---------------------- 4. Tag组柱状图（缩放后强区分度） ----------------------
fig, axes = plt.subplots(nrows=5, ncols=1, sharex=True, figsize=(18, 14))
fig.suptitle("Tag Layer Importance (Scaled for Contrast) - Layers 0-31", fontsize=16, y=0.96, fontweight="bold")

for idx, (key, color) in enumerate(zip(tag_keys, colors)):
    raw_vals = tag_data["raw"][key]    # 原始数值（用于标签）
    scaled_vals = tag_data["scaled"][key]  # 缩放后数值（用于绘图，放大差异）
    
    # 绘制缩放后的柱状图（高的更高、矮的更矮）
    bars = axes[idx].bar(layers, scaled_vals, color=color, alpha=0.8, 
                         edgecolor="black", linewidth=0.7, width=bar_width)
    
    # 固定y轴为 [-1.2, 1.2]（标准化范围+余量，确保所有组视觉统一）
    axes[idx].set_ylim(-1.2, 1.2)
    axes[idx].set_yticks([-1, -0.5, 0, 0.5, 1])  # 固定刻度，强化对比
    
    # 显示原始数值标签（缩放后仍能看到真实值）
    # for bar, raw_val, scaled_val in zip(bars, raw_vals, scaled_vals):
    #     va_align = "bottom" if scaled_val >= 0 else "top"
    #     offset = 0.05 if scaled_val >= 0 else -0.05  # 偏移量，避免与柱子重叠
    #     axes[idx].text(bar.get_x() + bar.get_width()/2., scaled_val + offset,
    #                   f"{raw_val:.1f}",
    #                   ha="center", va=va_align, fontsize=7.5, color="black", fontweight="medium")
    
    # 子图标签与网格（强化可读性）
    axes[idx].set_ylabel(f"{key.replace('_tag', '')}\n(Scaled to [-1,1])", 
                        fontsize=11, fontweight="medium", linespacing=1.2)
    axes[idx].grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.8)
    axes[idx].axhline(y=0, color="black", linewidth=0.5, linestyle="-", alpha=0.5)  # 添加y=0基准线

# 统一x轴
axes[-1].set_xlabel("Layer Index (0-31)", fontsize=12, fontweight="medium")
axes[-1].set_xticks(layers)
axes[-1].set_xticklabels(layers, rotation=0)

# 保存Tag组图
plt.tight_layout(rect=[0, 0.02, 1, 0.94])
tag_save_path = os.path.join(save_dir, "tag_layer_importance_high_contrast.png")
plt.savefig(tag_save_path, dpi=300, bbox_inches="tight")
plt.close()


# ---------------------- 5. Args组柱状图（同缩放逻辑） ----------------------
fig, axes = plt.subplots(nrows=5, ncols=1, sharex=True, figsize=(18, 14))
fig.suptitle("Args Layer Importance (Scaled for Contrast) - Layers 0-31", fontsize=16, y=0.96, fontweight="bold")

for idx, (key, color) in enumerate(zip(args_keys, colors)):
    raw_vals = args_data["raw"][key]
    scaled_vals = args_data["scaled"][key]
    
    # 绘制缩放后的柱状图
    bars = axes[idx].bar(layers, scaled_vals, color=color, alpha=0.8, 
                         edgecolor="black", linewidth=0.7, width=bar_width)
    
    # 固定y轴范围，统一视觉对比
    axes[idx].set_ylim(-1.2, 1.2)
    axes[idx].set_yticks([-1, -0.5, 0, 0.5, 1])
    
    # 显示原始数值标签
    # for bar, raw_val, scaled_val in zip(bars, raw_vals, scaled_vals):
    #     va_align = "bottom" if scaled_val >= 0 else "top"
    #     offset = 0.05 if scaled_val >= 0 else -0.05
    #     axes[idx].text(bar.get_x() + bar.get_width()/2., scaled_val + offset,
    #                   f"{raw_val:.1f}",
    #                   ha="center", va=va_align, fontsize=7.5, color="black", fontweight="medium")
    
    # 子图标签与网格
    axes[idx].set_ylabel(f"{key.replace('_arg', '')}\n(Scaled to [-1,1])", 
                        fontsize=11, fontweight="medium", linespacing=1.2)
    axes[idx].grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.8)
    axes[idx].axhline(y=0, color="black", linewidth=0.5, linestyle="-", alpha=0.5)

# 统一x轴
axes[-1].set_xlabel("Layer Index (0-31)", fontsize=12, fontweight="medium")
axes[-1].set_xticks(layers)
axes[-1].set_xticklabels(layers, rotation=0)

# 保存Args组图
plt.tight_layout(rect=[0, 0.02, 1, 0.94])
args_save_path = os.path.join(save_dir, "args_layer_importance_high_contrast.png")
plt.savefig(args_save_path, dpi=300, bbox_inches="tight")
plt.close()


# ---------------------- 6. 结果提示 ----------------------
print("强区分度可视化完成！图片保存路径：")
print(f"1. Tag组（高对比）：{tag_save_path}")
print(f"2. Args组（高对比）：{args_save_path}")
print("数据读取路径：", data_path)
    