"""Các biểu đồ khám phá và chuẩn bị dữ liệu khách hàng."""

from pathlib import Path
import matplotlib.pyplot as plt

from src.preprocessing import FEATURES


def plot_distributions(data, output_path):
    """Vẽ histogram ba đặc trưng và lưu biểu đồ vào đường dẫn được chỉ định."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    labels = ["Tuổi", "Thu nhập năm (đơn vị gốc)", "Điểm chi tiêu"]
    for ax, feature, label in zip(axes, FEATURES, labels):
        ax.hist(data[feature], bins=15, edgecolor="white")
        ax.set(xlabel=label, ylabel="Số khách hàng", title=label)
    fig.suptitle("Phân phối ba đặc trưng")
    fig.tight_layout()
    fig.savefig(Path(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scaling(data, scaled, output_path):
    """So sánh thang đo ban đầu và sau chuẩn hóa bằng biểu đồ hộp."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    labels = ["Tuổi", "Thu nhập", "Chi tiêu"]
    for ax, frame, title in zip(axes, (data, scaled), ("Trước chuẩn hóa", "Sau chuẩn hóa Z-score")):
        ax.boxplot([frame[col].dropna() for col in FEATURES], tick_labels=labels)
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(Path(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_split_sizes(splits, output_path):
    """Vẽ số khách hàng trong từng tập sau khi chia dữ liệu."""
    names = ["Train", "Validation", "Test"]
    counts = [len(splits[key]) for key in ("train", "validation", "test")]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, counts, color=["#3979b8", "#e99a3b", "#56a866"])
    ax.bar_label(bars)
    ax.set(title="Số khách hàng trong từng tập", ylabel="Số khách hàng", ylim=(0, max(counts) * 1.2))
    fig.tight_layout()
    fig.savefig(Path(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_kmeans_example(X, labels, centers, output_path):
    """Minh họa kết quả kiểm chứng K-means một chiều, không phải dữ liệu khách hàng."""
    fig, ax = plt.subplots(figsize=(8, 3))
    for cluster in range(len(centers)):
        points = X[labels == cluster, 0]
        ax.scatter(points, [cluster] * len(points), label=f"Cụm {cluster}", s=65)
        ax.scatter(centers[cluster, 0], cluster, marker="X", s=180, color="black")
    ax.set(xlabel="Giá trị ví dụ", ylabel="Mã cụm", title="Ví dụ kiểm chứng K-means một chiều")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_model_selection(results, chosen_k, output_path):
    """Vẽ Elbow trên train và Silhouette trên validation theo số cụm."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(results["k"], results["train_inertia"], marker="o")
    axes[0].set(title="Elbow trên train", xlabel="Số cụm K", ylabel="Inertia")
    axes[1].plot(results["k"], results["validation_silhouette"], marker="o")
    axes[1].axvline(chosen_k, color="red", linestyle="--", label=f"K được chọn = {chosen_k}")
    axes[1].set(title="Silhouette trên validation", xlabel="Số cụm K", ylabel="Silhouette")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(Path(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_clusters_3d(data, labels, output_path):
    """Vẽ cụm train trên ba trục nguyên gốc để mô tả khách hàng."""
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection="3d")
    for label in sorted(set(labels)):
        group = data.loc[labels == label]
        ax.scatter(group[FEATURES[0]], group[FEATURES[1]], group[FEATURES[2]], s=30, alpha=0.8, label=f"Cụm {label}")
    ax.set(xlabel="Tuổi", ylabel="Thu nhập năm (đơn vị gốc)", zlabel="",
           title="Phân cụm khách hàng tập train\nTrục đứng: điểm chi tiêu")
    ax.legend()
    ax.set_position([0.02, 0.05, 0.80, 0.85])
    fig.savefig(Path(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_candidate_review(data, runs, output_path):
    """Minh họa phân phối ba cột và kết quả khảo sát theo seed dữ liệu ứng viên."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, column in zip(axes[0], ["age", "income", "spending_score"]):
        ax.hist(data[column], bins=20, edgecolor="white")
        ax.set(title=column, ylabel="Số dòng")
    axes[1, 0].scatter(data["income"], data["spending_score"], alpha=0.35, s=9)
    axes[1, 0].set(xlabel="income", ylabel="spending_score")
    axes[1, 1].plot(runs["split_seed"].astype(str), runs["validation_silhouette"], marker="o", label="Validation")
    axes[1, 1].plot(runs["split_seed"].astype(str), runs["test_silhouette"], marker="o", label="Test khảo sát")
    axes[1, 1].set(xlabel="Seed chia tập", ylabel="Silhouette", ylim=(0, 1))
    axes[1, 1].legend()
    axes[1, 2].bar(runs["split_seed"].astype(str), runs["k"])
    axes[1, 2].set(xlabel="Seed chia tập", ylabel="K chọn từ validation", ylim=(0, 9))
    fig.suptitle("Khảo sát dữ liệu ứng viên 1.000 dòng")
    fig.tight_layout()
    fig.savefig(Path(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
