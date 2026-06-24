import json
import os
import matplotlib.pyplot as plt

result_dir = ".."
cats = ["phone", "Receipts", "StudentID", "military", "document", "Passport"]
prefixes = ["llava", "minigpt"]

for prefix in prefixes:
    var = None
    for cat in cats:
        with open(f"{result_dir}/{prefix}_sample_features_{cat}.json", "r") as f:
            var_t = json.load(f)
            # print(var_t)
        if var is None:
            # print(cat)
            var = var_t.copy()
        else:
            var = [v + v_t for v, v_t in zip(var, var_t)]
    var = [v/len(cats) for v in var]
    print(f"{prefix}: {var}")
    with open(f"{prefix}_sample_features_avg.json", "w") as f:
        json.dump(var, f, indent=4)

    layers = range(3, len(var) + 3)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "axes.linewidth": 0.8,
    })

    fig, ax = plt.subplots(figsize=(6.2, 4.0))

    # 冷色、低饱和、学术友好
    ax.plot(
        layers, var,
        label=prefix,
        color="#4C72B0",
        linewidth=2.0,
        marker="o",
        markersize=4,
        alpha=0.9
    )

    # ax.plot(
    #     layers, minigpt,
    #     label="MiniGPT-4",
    #     color="#55A868",
    #     linewidth=2.0,
    #     linestyle="--",
    #     marker="s",
    #     markersize=4,
    #     alpha=0.9
    # )

    ax.set_yscale("log")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Feature Variance")

    # 精简网格（只保留 y 轴主刻度）
    ax.grid(True, axis="y", which="major", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.grid(False, axis="x")

    # 去掉上右边框（高级感关键）
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ax.legend(frameon=False, loc="upper left")

    plt.tight_layout()
    plt.savefig(f"{prefix}_layer_variance.pdf", format="pdf")
    plt.show()