from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "result" / "figures" / "attention_shapley_20260809"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["Qwen3-8B", "Phi-3-mini-128k", "Llama3-8B"]
REGIONS = ["AUTH", "FACT", "ATTACK"]
REGION_KEYS = {"AUTH": "auth", "FACT": "data_fact", "ATTACK": "data_attack"}
LABELS = ["攻击成功", "攻击失败"]
METRICS = ["Attention总量", "单Token Attention", "Shapley贡献"]
METRIC_DISPLAY_NAMES = {
    "Attention总量": "Total Attention",
    "单Token Attention": "Per-token Attention",
    "Shapley贡献": "Shapley Contribution",
}
LABEL_DISPLAY_NAMES = {"攻击成功": "Attack succeeded", "攻击失败": "Attack failed"}
METRIC_COLORS = {
    "Attention总量": "#4C78A8",
    "单Token Attention": "#72A6C2",
    "Shapley贡献": "#D28E45",
}
REGION_COLORS = {"AUTH": "#5B8FB9", "FACT": "#7E8F7C", "ATTACK": "#C9675A"}

SOURCES = [
    {
        "dataset": "InjecAgent",
        "model": "Qwen3-8B",
        "attention": [
            "result/injecagent_qwen3_dh_attack_20260809/results.action_attention.jsonl",
            "result/injecagent_qwen3_dh_clean_20260809/results.action_attention.jsonl",
        ],
        "shapley": [
            "result/injecagent_qwen3_dh_attack_20260809/results.action_shapley.jsonl",
            "result/injecagent_qwen3_dh_clean_20260809/results.action_shapley.jsonl",
        ],
    },
    {
        "dataset": "InjecAgent",
        "model": "Phi-3-mini-128k",
        "attention": [
            "result/injecagent_phi3_dh_attack_20260809/results.action_attention.jsonl",
            "result/injecagent_phi3_dh_clean_20260809/results.action_attention.jsonl",
        ],
        "shapley": [
            "result/injecagent_phi3_dh_attack_20260809/results.action_shapley.jsonl",
            "result/injecagent_phi3_dh_clean_20260809/results.action_shapley.jsonl",
        ],
    },
    {
        "dataset": "InjecAgent",
        "model": "Llama3-8B",
        "attention": [
            "result/injecagent_llama3_dh_attack_20260810/results.action_attention.jsonl",
            "result/injecagent_llama3_dh_clean_20260810/results.action_attention.jsonl",
        ],
        "shapley": [
            "result/injecagent_llama3_dh_attack_20260810/results.action_shapley.jsonl",
            "result/injecagent_llama3_dh_clean_20260810/results.action_shapley.jsonl",
        ],
    },
    {
        "dataset": "AgentDojo",
        "model": "Qwen3-8B",
        "attention": ["result/agentdojo_fact_longer_than_attack_qwen_8b_chunk16_fixed/results.attention.jsonl"],
        "shapley": ["result/agentdojo_fact_longer_than_attack_qwen_8b_chunk16_fixed/results.shapley.jsonl"],
    },
    {
        "dataset": "AgentDojo",
        "model": "Phi-3-mini-128k",
        "attention": ["result/agentdojo_phi3_mini_128k_fixed/results.attention.jsonl"],
        "shapley": ["result/agentdojo_phi3_mini_128k_fixed/results.shapley.jsonl"],
    },
    {
        "dataset": "AgentDojo",
        "model": "Llama3-8B",
        "attention": ["result/agentdojo_llama3_8b/results.attention.jsonl"],
        "shapley": ["result/agentdojo_llama3_8b/results.shapley.jsonl"],
    },
]

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


def read_jsonl(paths: list[str]):
    for relative in paths:
        with (ROOT / relative).open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def label_for(dataset: str, row: dict) -> str:
    if dataset == "InjecAgent":
        return "攻击成功" if row.get("eval") == "succ" else "攻击失败"
    return "攻击成功" if row.get("security") is False else "攻击失败"


def span_length(token_ranges: dict, key: str) -> int:
    spans = (token_ranges or {}).get(key) or []
    if len(spans) == 2 and all(isinstance(value, (int, float)) for value in spans):
        spans = [spans]
    return sum(max(0, int(end) - int(start)) for start, end in spans)


def attention_scores(dataset: str, row: dict) -> dict:
    if dataset == "InjecAgent":
        return row.get("region_scores") or {}
    return row.get("region_scores_player_normalized") or {}


def build_rows() -> list[dict]:
    rows: list[dict] = []
    for source in SOURCES:
        dataset = source["dataset"]
        model = source["model"]
        for record in read_jsonl(source["attention"]):
            if dataset == "AgentDojo" and record.get("valid_for_stats") is not True:
                continue
            scores = attention_scores(dataset, record)
            label = label_for(dataset, record)
            for region, key in REGION_KEYS.items():
                length = span_length(record.get("token_ranges") or {}, key)
                score = scores.get(key)
                if length <= 0 or score is None or not math.isfinite(float(score)):
                    continue
                total = float(score)
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "label": label,
                        "region": region,
                        "metric": "Attention总量",
                        "length": length,
                        "value": total,
                    }
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "label": label,
                        "region": region,
                        "metric": "单Token Attention",
                        "length": length,
                        "value": total / length,
                    }
                )
        for record in read_jsonl(source["shapley"]):
            if record.get("valid_for_stats") is not True:
                continue
            label = label_for(dataset, record)
            for region, key in REGION_KEYS.items():
                length = span_length(record.get("token_ranges") or {}, key)
                score = record.get(f"phi_{key}")
                if length <= 0 or score is None or not math.isfinite(float(score)):
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "label": label,
                        "region": region,
                        "metric": "Shapley贡献",
                        "length": length,
                        "value": float(score),
                    }
                )
    return rows


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    rx = average_ranks(x)
    ry = average_ranks(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def group_rows(rows: list[dict]):
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["dataset"], row["model"], row["label"], row["region"], row["metric"])
        grouped[key].append(row)
    return grouped


def correlation_rows(grouped: dict[tuple, list[dict]]) -> list[dict]:
    output = []
    for key, records in sorted(grouped.items()):
        lengths = np.asarray([row["length"] for row in records], dtype=float)
        values = np.asarray([row["value"] for row in records], dtype=float)
        output.append(
            {
                "dataset": key[0],
                "model": key[1],
                "label": key[2],
                "region": key[3],
                "metric": key[4],
                "n": len(records),
                "unique_lengths": len(np.unique(lengths)),
                "spearman_rho": spearman(lengths, values),
            }
        )
    return output


def quartile_ids(lengths: np.ndarray) -> np.ndarray:
    ranks = average_ranks(lengths)
    percentiles = (ranks - 0.5) / len(lengths)
    return np.minimum(3, np.floor(percentiles * 4).astype(int))


def binned_rows(grouped: dict[tuple, list[dict]]) -> list[dict]:
    output = []
    for key, records in sorted(grouped.items()):
        lengths = np.asarray([row["length"] for row in records], dtype=float)
        values = np.asarray([row["value"] for row in records], dtype=float)
        bins = quartile_ids(lengths)
        mean = float(np.mean(values))
        std = float(np.std(values))
        zvalues = (values - mean) / std if std > 0 else np.zeros_like(values)
        for bin_id in range(4):
            mask = bins == bin_id
            if not np.any(mask):
                continue
            output.append(
                {
                    "dataset": key[0],
                    "model": key[1],
                    "label": key[2],
                    "region": key[3],
                    "metric": key[4],
                    "length_bin": f"Q{bin_id + 1}",
                    "n": int(np.sum(mask)),
                    "length_min": int(np.min(lengths[mask])),
                    "length_max": int(np.max(lengths[mask])),
                    "length_mean": float(np.mean(lengths[mask])),
                    "value_mean": float(np.mean(values[mask])),
                    "value_std": float(np.std(values[mask])),
                    "standardized_mean": float(np.mean(zvalues[mask])),
                }
            )
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig: mpl.figure.Figure, stem: str) -> None:
    for suffix, kwargs in {
        "png": {"dpi": 300},
        "tiff": {"dpi": 600},
        "svg": {},
        "pdf": {},
    }.items():
        fig.savefig(OUT / f"{stem}.{suffix}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


def plot_correlations(correlations: list[dict]) -> None:
    row_keys = [(dataset, model, label) for dataset in ["InjecAgent", "AgentDojo"] for model in MODELS for label in LABELS]
    row_labels = [f"{dataset} · {model}\n{label}" for dataset, model, label in row_keys]
    row_labels = [
        f"{dataset} · {model}\n{LABEL_DISPLAY_NAMES[label]}"
        for dataset, model, label in row_keys
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 5.8), sharey=True)
    lookup = {
        (row["dataset"], row["model"], row["label"], row["region"], row["metric"]): row
        for row in correlations
    }
    image = None
    for panel, (ax, metric) in enumerate(zip(axes, METRICS)):
        matrix = np.full((len(row_keys), len(REGIONS)), np.nan)
        for i, row_key in enumerate(row_keys):
            for j, region in enumerate(REGIONS):
                record = lookup.get((*row_key, region, metric))
                if record:
                    matrix[i, j] = record["spearman_rho"]
        masked = np.ma.masked_invalid(matrix)
        image = ax.imshow(masked, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if math.isfinite(matrix[i, j]):
                    color = "white" if abs(matrix[i, j]) >= 0.55 else "#202020"
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=6, color=color)
                else:
                    ax.text(j, i, "NA", ha="center", va="center", fontsize=5.5, color="#777777")
        ax.set_xticks(range(len(REGIONS)), REGIONS)
        ax.set_title(METRIC_DISPLAY_NAMES[metric])
        ax.tick_params(length=0)
        for y in [1.5, 3.5, 5.5, 7.5, 9.5]:
            ax.axhline(y, color="white", linewidth=1.5)
        ax.text(-0.12, 1.03, "abc"[panel], transform=ax.transAxes, fontsize=8, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_yticks(range(len(row_labels)), row_labels)
    cbar = fig.colorbar(image, ax=axes, fraction=0.018, pad=0.025)
    cbar.set_label("区域长度与指标的 Spearman ρ")
    fig.suptitle("区域长度与 Attention/Shapley 的分层相关性", fontsize=9, y=0.985)
    cbar.set_label("Spearman's ρ between Region Length and Metric")
    fig.suptitle(
        "Stratified Correlations between Region Length and Attention/Shapley",
        fontsize=9,
        y=0.985,
    )
    fig.subplots_adjust(left=0.23, right=0.91, top=0.91, bottom=0.08, wspace=0.14)
    save_figure(fig, "fig6_region_length_correlations")


def length_reference_rows(bins: list[dict]) -> list[dict]:
    """Summarize the concrete token scale represented by each quartile.

    Quartiles are defined within model × attack-label strata.  For a compact
    manuscript table, report the median of the six stratum-specific mean
    lengths (three models × two labels) for each dataset × region × quartile.
    """
    attention_bins = [row for row in bins if row["metric"] == "Attention总量"]
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for row in attention_bins:
        grouped[(row["dataset"], row["region"], row["length_bin"])].append(row["length_mean"])
    output = []
    for dataset in ["InjecAgent", "AgentDojo"]:
        for region in REGIONS:
            record = {"dataset": dataset, "region": region}
            for bin_id in range(4):
                label = f"Q{bin_id + 1}"
                record[label] = float(np.median(grouped[(dataset, region, label)]))
            output.append(record)
    return output


def plot_binned_trends(bins: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.8), sharey=True)
    for ax in axes[1:]:
        ax.sharey(axes[0])
    x = np.arange(4)
    for panel, (ax, metric) in enumerate(zip(axes, METRICS)):
        for region in REGIONS:
            means = []
            errors = []
            for bin_id in range(4):
                label = f"Q{bin_id + 1}"
                values = [
                    row["standardized_mean"]
                    for row in bins
                    if row["metric"] == metric and row["region"] == region and row["length_bin"] == label
                ]
                means.append(float(np.mean(values)) if values else np.nan)
                errors.append(float(np.std(values) / math.sqrt(len(values))) if len(values) > 1 else 0.0)
            ax.errorbar(
                x,
                means,
                yerr=errors,
                marker="o",
                markersize=4,
                linewidth=1.5,
                capsize=2.5,
                color=REGION_COLORS[region],
                label=region,
            )
        ax.axhline(0, color="#777777", linewidth=0.7)
        ax.set_xticks(x, ["Q1\n最短", "Q2", "Q3", "Q4\n最长"])
        ax.set_title(metric)
        ax.set_xlabel("分层内区域长度四分位")
        ax.grid(axis="y", color="#E3E3E3", linewidth=0.5)
        ax.text(-0.12, 1.04, "abc"[panel], transform=ax.transAxes, fontsize=8, fontweight="bold")
    axes[0].set_ylabel("分层内标准化指标均值（z-score）")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[2].legend(handles, labels, loc="upper right", ncol=1)

    fig.suptitle("区域长度分箱下的 Attention 与 Shapley 变化", fontsize=9, y=0.985)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.24, wspace=0.18)
    save_figure(fig, "fig7_region_length_bins")


def plot_binned_trends_by_dataset(bins: list[dict]) -> None:
    """Plot the Figure 7 trends separately for each dataset."""
    datasets = ["InjecAgent", "AgentDojo"]
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.2), sharex=True, sharey=True)
    x = np.arange(4)
    panel_labels = "abcdef"

    for row_index, dataset in enumerate(datasets):
        for column_index, metric in enumerate(METRICS):
            ax = axes[row_index, column_index]
            for region in REGIONS:
                means = []
                errors = []
                for bin_id in range(4):
                    label = f"Q{bin_id + 1}"
                    values = [
                        row["standardized_mean"]
                        for row in bins
                        if row["dataset"] == dataset
                        and row["metric"] == metric
                        and row["region"] == region
                        and row["length_bin"] == label
                    ]
                    means.append(float(np.mean(values)) if values else np.nan)
                    errors.append(
                        float(np.std(values) / math.sqrt(len(values)))
                        if len(values) > 1
                        else 0.0
                    )
                ax.errorbar(
                    x,
                    means,
                    yerr=errors,
                    marker="o",
                    markersize=3.8,
                    linewidth=1.4,
                    capsize=2.3,
                    color=REGION_COLORS[region],
                    label=region,
                )
            ax.axhline(0, color="#777777", linewidth=0.7)
            ax.set_xticks(x, ["Q1\n最短", "Q2", "Q3", "Q4\n最长"])
            ax.grid(axis="y", color="#E3E3E3", linewidth=0.5)
            ax.text(
                -0.12,
                1.04,
                panel_labels[row_index * 3 + column_index],
                transform=ax.transAxes,
                fontsize=8,
                fontweight="bold",
            )
            if row_index == 0:
                ax.set_title(metric)
            if column_index == 0:
                ax.set_ylabel(f"{dataset}\n标准化指标均值（z-score）")
            if row_index == 1:
                ax.set_xlabel("区域长度四分位")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 2].legend(handles, labels, loc="upper right", ncol=1)
    fig.suptitle("不同数据集的区域长度分箱趋势", fontsize=9, y=0.985)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.91, bottom=0.13, hspace=0.30, wspace=0.16)
    save_figure(fig, "fig8_region_length_bins_by_dataset")


def summary_rows(correlations: list[dict]) -> list[dict]:
    output = []
    for metric in METRICS:
        for region in REGIONS + ["ALL"]:
            values = [
                row["spearman_rho"]
                for row in correlations
                if row["metric"] == metric
                and (region == "ALL" or row["region"] == region)
                and math.isfinite(row["spearman_rho"])
            ]
            array = np.asarray(values, dtype=float)
            output.append(
                {
                    "metric": metric,
                    "region": region,
                    "strata": len(values),
                    "median_rho": float(np.median(array)) if len(array) else float("nan"),
                    "q1_rho": float(np.quantile(array, 0.25)) if len(array) else float("nan"),
                    "q3_rho": float(np.quantile(array, 0.75)) if len(array) else float("nan"),
                    "positive_strata": int(np.sum(array > 0)) if len(array) else 0,
                    "negative_strata": int(np.sum(array < 0)) if len(array) else 0,
                }
            )
    return output


def main() -> None:
    rows = build_rows()
    grouped = group_rows(rows)
    correlations = correlation_rows(grouped)
    bins = binned_rows(grouped)
    summaries = summary_rows(correlations)
    length_references = length_reference_rows(bins)
    write_csv(OUT / "fig6_region_length_correlations.csv", correlations)
    write_csv(OUT / "fig7_region_length_bins.csv", bins)
    write_csv(OUT / "fig7_region_length_reference.csv", length_references)
    write_csv(OUT / "fig8_region_length_bins_by_dataset.csv", bins)
    write_csv(OUT / "region_length_effect_summary.csv", summaries)
    plot_binned_trends(bins)
    plot_binned_trends_by_dataset(bins)
    print(f"Long rows: {len(rows)}")
    print(f"Correlation strata: {len(correlations)}")
    print(f"Binned strata: {len(bins)}")
    print(f"Outputs written to {OUT}")


if __name__ == "__main__":
    main()
