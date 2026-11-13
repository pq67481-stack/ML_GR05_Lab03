from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .evaluate import METRICS
from .preprocess import AGE, INCOME, SPEND

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "axes.labelcolor": "#223246",
    "figure.facecolor": "#ffffff",
    "savefig.dpi": 150,
})
COLORS = list(plt.get_cmap("tab10").colors)
FAMILY_NAMES = {"kmeans": "K-means", "gmm": "Gaussian Mixture"}


def save(figure, path: Path) -> None:
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_eda(clean: pd.DataFrame, features: list[str], folder: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    for ax, feature in zip(axes[0], features, strict=True):
        ax.hist(clean[feature].dropna(), bins=12, color="#397a87", edgecolor="white")
        ax.set(xlabel=feature, ylabel="Số khách")
    for ax, (x, y) in zip(axes[1], [(INCOME, SPEND), (AGE, SPEND), (AGE, INCOME)], strict=True):
        ax.scatter(clean[x], clean[y], s=17, color="#245a80", alpha=0.65)
        ax.set(xlabel=x, ylabel=y)
    fig.suptitle("Phân phối và quan hệ giữa các biến đầu vào", fontsize=16, fontweight="bold")
    save(fig, folder / "eda.png")


def plot_search(scores: pd.DataFrame, folder: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), layout="constrained")
    titles = ("Silhouette ↑", "Davies–Bouldin ↓", "Calinski–Harabasz ↑")
    for ax, metric, title in zip(axes, METRICS, titles, strict=True):
        for color, (family, subset) in zip(COLORS, scores.groupby("family")):
            subset = subset.sort_values("k")
            ax.errorbar(subset["k"], subset[f"{metric}_mean"], yerr=subset[f"{metric}_std"],
                        marker="o", capsize=3, color=color, label=FAMILY_NAMES[family])
        ax.set(xlabel="Số cụm / thành phần", title=title)
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)
    fig.suptitle("Kiểm định chéo trên tập train · Trung bình ± độ lệch chuẩn", fontsize=15)
    save(fig, folder / "cv_metrics.png")


def plot_segments(train: pd.DataFrame, test: pd.DataFrame, folder: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for ax, x in zip(axes, [INCOME, AGE], strict=True):
        for label, rows in train.groupby("cluster"):
            color = COLORS[int(label) % len(COLORS)]
            ax.scatter(rows[x], rows[SPEND], color=color, s=23, alpha=0.75, label=f"Cụm {label}")
            test_rows = test.loc[test["cluster"].eq(label)]
            ax.scatter(test_rows[x], test_rows[SPEND], edgecolor=color, facecolor="none",
                       s=55, linewidth=1.2)
        ax.set(xlabel=x, ylabel=SPEND)
    axes[1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    fig.suptitle("Phân cụm khách hàng · Train tô màu, test viền rỗng", fontsize=15)
    save(fig, folder / "segments.png")


def plot_silhouette(train: pd.DataFrame, test: pd.DataFrame, folder: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, rows, title in zip(axes, [train, test], ["Train", "Test"], strict=True):
        y = 0
        ticks, labels = [], []
        for label, subset in rows.groupby("cluster"):
            values = np.sort(subset["silhouette"].to_numpy())
            ax.barh(np.arange(y, y + len(values)), values, height=1,
                    color=COLORS[int(label) % len(COLORS)])
            ticks.append(y + (len(values) - 1) / 2)
            labels.append(f"Cụm {label} · {len(values)}")
            y += len(values) + 2
        average = rows["silhouette"].mean()
        if pd.notna(average):
            ax.axvline(average, color="#1c2735", linestyle="--", label=f"TB {average:.3f}")
            ax.legend(frameon=False)
        ax.axvline(0, color="#a6adb5", linewidth=0.9)
        ax.set(xlim=(-1, 1), yticks=ticks, yticklabels=labels, title=title, xlabel="Silhouette từng khách")
    save(fig, folder / "silhouette.png")
