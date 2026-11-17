"""Vẽ ảnh gốc và kết quả phân cụm màu RGB."""

from pathlib import Path

import matplotlib.pyplot as plt


def plot_comparison(original, results, output_path):
    """Lưu hình so sánh ảnh gốc và từng ảnh phân vùng, theo đúng thứ tự K."""
    fig, axes = plt.subplots(1, len(results) + 1, figsize=(4 * (len(results) + 1), 5))
    axes[0].imshow(original)
    axes[0].set_title("Ảnh gốc")
    axes[0].axis("off")
    for ax, (k, image) in zip(axes[1:], results.items()):
        ax.imshow(image)
        ax.set_title(f"K = {k}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(Path(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
