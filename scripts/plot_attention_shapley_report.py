from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "result" / "figures" / "attention_shapley_20260809"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "legend.frameon": False,
    }
)

METHODS = ["Attention", "Shapley", "Shapley+LLM"]
METHOD_COLORS = {
    "Attention": "#709BC4",
    "Shapley": "#E3A55D",
    "Shapley+LLM": "#6AAA8B",
}
MODELS = ["Qwen3-8B", "Phi-3-mini-128k", "Llama3-8B"]
MODEL_COLORS = {
    "Qwen3-8B": "#4C78A8",
    "Phi-3-mini-128k": "#D28E45",
    "Llama3-8B": "#8C6BB1",
}
LABEL_COLORS = {"攻击成功": "#C9675A", "攻击失败": "#5B8FB9"}
LABEL_DISPLAY_NAMES = {"攻击成功": "Attack succeeded", "攻击失败": "Attack failed"}
DATASET_LINESTYLES = {"InjecAgent": "-", "AgentDojo": "--"}


RESULTS = [
    ("InjecAgent", "Qwen3-8B", "Attention", 510, 481, 29, 0, 52.84, 51.46, 100.00, 94.31, 67.96),
    ("InjecAgent", "Qwen3-8B", "Shapley", 478, 264, 246, 32, 70.98, 64.42, 93.73, 51.76, 76.28),
    ("InjecAgent", "Qwen3-8B", "Shapley+LLM", 474, 2, 508, 36, 96.27, 99.58, 92.94, 0.39, 96.14),
    ("InjecAgent", "Phi-3-mini-128k", "Attention", 121, 51, 459, 389, 56.86, 70.35, 23.73, 10.00, 35.48),
    ("InjecAgent", "Phi-3-mini-128k", "Shapley", 509, 301, 209, 1, 70.39, 62.84, 99.80, 59.02, 77.12),
    ("InjecAgent", "Phi-3-mini-128k", "Shapley+LLM", 505, 2, 508, 5, 99.31, 99.61, 99.02, 0.39, 99.31),
    ("InjecAgent", "Llama3-8B", "Attention", 510, 463, 47, 0, 54.61, 52.42, 100.00, 90.78, 68.78),
    ("InjecAgent", "Llama3-8B", "Shapley", 375, 244, 266, 135, 62.84, 60.58, 73.53, 47.84, 66.43),
    ("InjecAgent", "Llama3-8B", "Shapley+LLM", 373, 2, 508, 137, 86.37, 99.47, 73.14, 0.39, 84.29),
    ("AgentDojo", "Qwen3-8B", "Attention", 135, 124, 32, 1, 57.19, 52.12, 99.26, 79.49, 68.35),
    ("AgentDojo", "Qwen3-8B", "Shapley", 73, 45, 111, 63, 63.01, 61.86, 53.68, 28.85, 57.48),
    ("AgentDojo", "Qwen3-8B", "Shapley+LLM", 73, 14, 142, 63, 73.63, 83.91, 53.68, 8.97, 65.47),
    ("AgentDojo", "Phi-3-mini-128k", "Attention", 134, 131, 25, 2, 54.45, 50.57, 98.53, 83.97, 66.83),
    ("AgentDojo", "Phi-3-mini-128k", "Shapley", 57, 40, 116, 79, 59.25, 58.76, 41.91, 25.64, 48.93),
    ("AgentDojo", "Phi-3-mini-128k", "Shapley+LLM", 56, 17, 139, 80, 66.78, 76.71, 41.18, 10.90, 53.59),
    ("AgentDojo", "Llama3-8B", "Attention", 135, 128, 28, 1, 55.82, 51.33, 99.26, 82.05, 67.67),
    ("AgentDojo", "Llama3-8B", "Shapley", 60, 42, 114, 76, 59.59, 58.82, 44.12, 26.92, 50.42),
    ("AgentDojo", "Llama3-8B", "Shapley+LLM", 54, 19, 137, 82, 65.41, 73.97, 39.71, 12.18, 51.67),
]

ATTENTION = [
    ("InjecAgent", "Qwen3-8B", "攻击成功", 0.032355, 0.088149, 0.018382),
    ("InjecAgent", "Qwen3-8B", "攻击失败", 0.039465, 0.066956, 0.011540),
    ("InjecAgent", "Phi-3-mini-128k", "攻击成功", 0.021003, 0.016763, 0.001983),
    ("InjecAgent", "Phi-3-mini-128k", "攻击失败", 0.036927, 0.011461, 0.001588),
    ("InjecAgent", "Llama3-8B", "攻击成功", 0.019409, 0.060643, 0.003111),
    ("InjecAgent", "Llama3-8B", "攻击失败", 0.028565, 0.036319, 0.003015),
    ("AgentDojo", "Qwen3-8B", "攻击成功", 0.134785, 0.524079, 0.341136),
    ("AgentDojo", "Qwen3-8B", "攻击失败", 0.201522, 0.534033, 0.264444),
    ("AgentDojo", "Phi-3-mini-128k", "攻击成功", 0.086141, 0.492528, 0.421331),
    ("AgentDojo", "Phi-3-mini-128k", "攻击失败", 0.112107, 0.520555, 0.367338),
    ("AgentDojo", "Llama3-8B", "攻击成功", 0.086862, 0.509240, 0.403897),
    ("AgentDojo", "Llama3-8B", "攻击失败", 0.123297, 0.530571, 0.346132),
]

SHAPLEY = [
    ("InjecAgent", "Qwen3-8B", "攻击成功", -0.027172, 0.127363, 1.247751),
    ("InjecAgent", "Qwen3-8B", "攻击失败", 0.040500, 0.725387, 0.822047),
    ("InjecAgent", "Phi-3-mini-128k", "攻击成功", -0.040577, -0.111029, 0.689237),
    ("InjecAgent", "Phi-3-mini-128k", "攻击失败", 0.018592, 0.330590, 0.460370),
    ("InjecAgent", "Llama3-8B", "攻击成功", -0.307991, 0.762391, 1.087711),
    ("InjecAgent", "Llama3-8B", "攻击失败", -0.117002, 0.669300, 0.657206),
    ("AgentDojo", "Qwen3-8B", "攻击成功", 0.3746, 0.2883, 1.0622),
    ("AgentDojo", "Qwen3-8B", "攻击失败", 0.6335, 1.2554, 0.6482),
    ("AgentDojo", "Phi-3-mini-128k", "攻击成功", -0.058895, 0.406608, 0.487896),
    ("AgentDojo", "Phi-3-mini-128k", "攻击失败", 0.207302, 0.541881, 0.122950),
    ("AgentDojo", "Llama3-8B", "攻击成功", 0.112659, 0.505044, 0.628866),
    ("AgentDojo", "Llama3-8B", "攻击失败", 0.267324, 1.615974, 0.060220),
]

TIMES = [
    ("InjecAgent", "Qwen3-8B", 1.349, 0.680, 4.219),
    ("InjecAgent", "Phi-3-mini-128k", 1.076, 0.519, 3.343),
    ("InjecAgent", "Llama3-8B", 1.217, 0.627, 3.648),
    ("AgentDojo", "Qwen3-8B", 5.519, 1.748, 29.444),
    ("AgentDojo", "Phi-3-mini-128k", 6.015, 1.155, 35.549),
    ("AgentDojo", "Llama3-8B", 4.288, 1.218, 24.146),
]


def save_figure(fig: mpl.figure.Figure, stem: str) -> None:
    for suffix, kwargs in {
        "png": {"dpi": 300},
        "tiff": {"dpi": 600},
        "svg": {},
        "pdf": {},
    }.items():
        fig.savefig(OUT / f"{stem}.{suffix}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


def write_csv(name: str, header: list[str], rows: list[tuple]) -> None:
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def add_panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(-0.13, 1.06, label, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")


def plot_metric_heatmap() -> None:
    metrics = ["Accuracy", "Precision", "Recall", "FPR", "F1"]
    actual = np.array([r[7:] for r in RESULTS], dtype=float)
    desirability = actual.copy()
    desirability[:, 3] = 100 - desirability[:, 3]
    labels = [f"{r[0]} · {r[1]} · {r[2]}" for r in RESULTS]

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    image = ax.imshow(desirability, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
    for i in range(actual.shape[0]):
        for j in range(actual.shape[1]):
            color = "white" if desirability[i, j] > 66 else "#202020"
            ax.text(j, i, f"{actual[i, j]:.1f}", ha="center", va="center", fontsize=6.1, color=color)
    ax.set_xticks(range(len(metrics)), ["Accuracy↑", "Precision↑", "Recall↑", "FPR↓", "F1↑"])
    ax.set_yticks(range(len(labels)), labels)
    ax.tick_params(length=0)
    for y in [2.5, 5.5, 8.5, 11.5, 14.5]:
        ax.axhline(y, color="white", linewidth=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("性能方向统一后的得分（越高越好）")
    ax.set_title("三种判定方法的整体性能", pad=8)
    fig.subplots_adjust(left=0.37, right=0.92, top=0.92, bottom=0.10)
    save_figure(fig, "fig1_metric_heatmap")


def plot_error_progression() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharex=True)
    for dataset in ["InjecAgent", "AgentDojo"]:
        for model in MODELS:
            rows = [r for r in RESULTS if r[0] == dataset and r[1] == model]
            rows.sort(key=lambda r: METHODS.index(r[2]))
            label = f"{dataset} · {model}"
            style = DATASET_LINESTYLES[dataset]
            color = MODEL_COLORS[model]
            axes[0].plot(METHODS, [r[4] for r in rows], marker="o", ms=4, lw=1.7, ls=style, color=color, label=label)
            axes[1].plot(METHODS, [r[6] for r in rows], marker="o", ms=4, lw=1.7, ls=style, color=color, label=label)
    axes[0].set_title("假阳性（FP）")
    axes[1].set_title("假阴性（FN）")
    for ax in axes:
        ax.set_ylabel("样本数")
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.5)
        ax.tick_params(axis="x", rotation=15)
    add_panel_label(axes[0], "a")
    add_panel_label(axes[1], "b")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=3)
    fig.suptitle("判定条件收紧时的错误类型变化", y=1.01, fontsize=9)
    fig.subplots_adjust(bottom=0.32, wspace=0.34)
    save_figure(fig, "fig2_error_progression")


def plot_grouped_regions(data: list[tuple], stem: str, title: str, ylabel: str, include_zero: bool) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.0))
    panels = [
        ("InjecAgent", "Qwen3-8B"),
        ("InjecAgent", "Phi-3-mini-128k"),
        ("InjecAgent", "Llama3-8B"),
        ("AgentDojo", "Qwen3-8B"),
        ("AgentDojo", "Phi-3-mini-128k"),
        ("AgentDojo", "Llama3-8B"),
    ]
    x = np.arange(3)
    width = 0.34
    for panel_index, (ax, (dataset, model)) in enumerate(zip(axes.flat, panels)):
        rows = {(r[2]): r[3:] for r in data if r[0] == dataset and r[1] == model}
        for offset, label in [(-width / 2, "攻击成功"), (width / 2, "攻击失败")]:
            ax.bar(
                x + offset,
                rows[label],
                width,
                color=LABEL_COLORS[label],
                label=LABEL_DISPLAY_NAMES[label],
                edgecolor="white",
                linewidth=0.4,
            )
        if include_zero:
            ax.axhline(0, color="#666666", lw=0.6)
        ax.set_xticks(x, ["AUTH", "FACT", "ATTACK"])
        ax.set_title(f"{dataset} · {model}")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#E3E3E3", linewidth=0.5)
        add_panel_label(ax, "abcdef"[panel_index])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(title, y=1.01, fontsize=9)
    fig.subplots_adjust(bottom=0.13, hspace=0.46, wspace=0.30)
    save_figure(fig, stem)


def plot_efficiency() -> None:
    labels = [f"{r[0]}\n{r[1]}" for r in TIMES]
    means = np.array([r[2] for r in TIMES])
    lows = means - np.array([r[3] for r in TIMES])
    highs = np.array([r[4] for r in TIMES]) - means
    colors = [MODEL_COLORS[r[1]] for r in TIMES]
    x = np.arange(len(TIMES))

    fig, ax = plt.subplots(figsize=(9.0, 2.8))
    ax.errorbar(x, means, yerr=[lows, highs], fmt="none", ecolor="#777777", elinewidth=1.2, capsize=3, zorder=1)
    ax.scatter(x, means, s=42, c=colors, edgecolor="white", linewidth=0.7, zorder=2)
    for xi, val in zip(x, means):
        ax.text(xi, val * 1.16, f"{val:.2f}s", ha="center", va="bottom", fontsize=6.5)
    ax.set_yscale("log")
    ax.set_xticks(x, labels)
    ax.set_ylabel("单样本耗时（秒，对数刻度）")
    ax.set_title("Shapley计算耗时：圆点为均值，线段为最短—最长范围")
    ax.grid(axis="y", which="both", color="#E0E0E0", linewidth=0.5)
    fig.subplots_adjust(bottom=0.24, top=0.85)
    save_figure(fig, "fig5_shapley_efficiency")


def main() -> None:
    write_csv(
        "fig1_fig2_main_results.csv",
        ["dataset", "model", "method", "TP", "FP", "TN", "FN", "accuracy_pct", "precision_pct", "recall_pct", "fpr_pct", "f1_pct"],
        RESULTS,
    )
    write_csv("fig3_attention_means.csv", ["dataset", "model", "label", "AUTH", "FACT", "ATTACK"], ATTENTION)
    write_csv("fig4_shapley_means.csv", ["dataset", "model", "label", "AUTH", "FACT", "ATTACK"], SHAPLEY)
    write_csv("fig5_shapley_time.csv", ["dataset", "model", "mean_seconds", "min_seconds", "max_seconds"], TIMES)
    plot_metric_heatmap()
    plot_error_progression()
    plot_grouped_regions(
        ATTENTION,
        "fig3_attention_attribution",
        "Regional Attention by Attack Outcome",
        "Mean Attention",
        False,
    )
    plot_grouped_regions(
        SHAPLEY,
        "fig4_shapley_attribution",
        "Regional Shapley Contributions by Attack Outcome",
        "Mean Shapley Contribution",
        True,
    )
    plot_efficiency()
    print(f"Figures written to {OUT}")


if __name__ == "__main__":
    main()
