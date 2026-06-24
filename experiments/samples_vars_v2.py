import json
import os
import matplotlib.pyplot as plt
import numpy as np

# ===================== 基础配置 =====================
# 颜色配置：6个隐私主体浅色系，平均值深色主色
CAT_COLORS = {
    "phone": "#92C5DE",  # 浅蓝
    "Receipts": "#F4A582",  # 浅橙
    "StudentID": "#66C2A5",  # 浅绿
    "military": "#FC8D62",  # 浅红
    "document": "#8DA0CB",  # 浅紫
    "Passport": "#E78AC3",  # 浅粉
    "avg": "#4C72B0"  # 深蓝（突出平均值）
}

# 文件路径与类别配置（完全匹配你的保存逻辑）
result_dir = ".."  # 当前目录
cats = ["phone", "Receipts", "StudentID", "military", "document", "Passport"]
prefixes = ["llava", "minigpt"]  # 模型前缀

# ===================== 核心绘图逻辑 =====================
for prefix in prefixes:
    # 初始化数据存储
    cat_means = {}  # 各类别均值
    cat_stds = {}  # 各类别标准差（方差开根号）
    avg_mean = None  # 所有类别均值的平均值
    avg_std = None  # 所有类别标准差的平均值

    # 1. 读取stats文件，计算均值+标准差
    for cat in cats:
        try:
            # 匹配保存的stats文件名
            stats_path = os.path.join(result_dir, f"{prefix}_sample_stats_{cat}.json")
            if not os.path.exists(stats_path):
                print(f"警告：{stats_path} 不存在，跳过{cat}")
                continue

            # 读取stats文件（均值/方差）
            with open(stats_path, "r") as f:
                stats_data = json.load(f)
            mean_t = stats_data["layer_mean"]  # 均值
            var_t = stats_data["layer_variance"]  # 方差

            # 方差→标准差（核心转换）
            std_t = np.sqrt(np.array(var_t)).tolist()

            # 校验数据长度（模型层3-19，共17层）
            if len(mean_t) != 17 or len(std_t) != 17:
                print(f"警告：{prefix}-{cat} 数据长度异常（应为17层），跳过")
                continue

            # 存储数据
            cat_means[cat] = mean_t
            cat_stds[cat] = std_t
            print(f"✅ 读取成功：{prefix}-{cat} | 均值范围：{min(mean_t):.2f}~{max(mean_t):.2f}")

        except Exception as e:
            print(f"错误：处理{prefix}-{cat}失败 - {str(e)}")
            continue

    # 2. 过滤有效数据，计算平均值
    valid_cats = [c for c in cats if c in cat_means and c in cat_stds]
    if not valid_cats:
        print(f"❌ {prefix} 无有效数据，跳过绘图")
        continue

    # 统一维度（已校验为17层，直接计算平均）
    mean_matrix = np.array([cat_means[cat] for cat in valid_cats])
    std_matrix = np.array([cat_stds[cat] for cat in valid_cats])
    avg_mean = np.mean(mean_matrix, axis=0).tolist()
    avg_std = np.mean(std_matrix, axis=0).tolist()

    # 3. 绘制均值±标准差图（论文级样式）
    layers = range(3, 20)  # X轴：模型层3-19
    plt.rcParams.update({
        "font.family": "serif",  # 学术衬线字体
        "font.size": 14,  # 基础字体大小
        "axes.labelsize": 16,  # 坐标轴标签大小
        "legend.fontsize": 10,  # 图例字体
        "axes.linewidth": 1.5,  # 坐标轴边框粗细
        "xtick.labelsize": 12,  # X轴刻度字体
        "ytick.labelsize": 12,  # Y轴刻度字体
        "figure.dpi": 300,  # 高清分辨率
        "figure.facecolor": "white"  # 白底（适配论文排版）
    })

    # 创建画布
    fig, ax = plt.subplots(figsize=(8, 4.5))

    # 绘制各隐私类别的均值线+标准差阴影（浅色次要视觉）
    for cat in valid_cats:
        mean = np.array(cat_means[cat])
        std = np.array(cat_stds[cat])
        color = CAT_COLORS[cat]

        # 均值折线（细线条）
        ax.plot(layers, mean, color=color, linewidth=1.5, label=cat, alpha=0.7)
        # 标准差阴影（半透明，体现波动范围）
        ax.fill_between(layers, mean - std, mean + std, color=color, alpha=0.2)

    # 绘制平均值的均值线+标准差阴影（深色突出视觉）
    avg_mean_np = np.array(avg_mean)
    avg_std_np = np.array(avg_std)
    ax.plot(
        layers, avg_mean_np,
        color=CAT_COLORS["avg"], linewidth=3.5, marker="o", markersize=5,
        label="Average", alpha=0.9
    )
    ax.fill_between(
        layers, avg_mean_np - avg_std_np, avg_mean_np + avg_std_np,
        color=CAT_COLORS["avg"], alpha=0.3
    )

    # 图表样式优化
    ax.set_xlabel("Layer")  # X轴标签
    ax.set_ylabel("Feature Mean ± Std")  # Y轴标签（核心：均值±标准差）
    ax.grid(True, which="major", linestyle="--", alpha=0.6)  # 主网格
    ax.spines["top"].set_visible(False)  # 隐藏上边框
    ax.spines["right"].set_visible(False)  # 隐藏右边框
    ax.legend(frameon=False, loc="upper left", ncol=2)  # 图例（无框，分两列）

    # 紧凑布局+保存
    plt.tight_layout()
    save_path = f"{prefix}_layer_mean_std.pdf"
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    print(f"✅ {prefix} 均值±标准差图已保存：{save_path}")
    plt.show()