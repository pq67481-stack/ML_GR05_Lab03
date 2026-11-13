import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.cbook import get_sample_data
from PIL import Image, ImageOps
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from .preprocess import ROOT
from .visualize import plt, save


def segment_image(path: Path | None = None, cluster_counts=(3, 6, 9), max_side: int = 320) -> pd.DataFrame:
    if max_side < 2 or not cluster_counts or len(set(cluster_counts)) != len(cluster_counts):
        raise ValueError("Kích thước cạnh phải từ 2 và danh sách K phải gồm các giá trị khác nhau.")
    source = path or Path(get_sample_data("grace_hopper.jpg", asfileobj=False))
    with Image.open(source) as opened:
        original = ImageOps.exif_transpose(opened).convert("RGB")
    resized = original.copy()
    resized.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    width, height = resized.size
    pixels = np.asarray(resized, dtype=np.float32).reshape(-1, 3)
    unique_colors = len(np.unique(pixels, axis=0))
    if min(cluster_counts) < 2 or max(cluster_counts) > unique_colors:
        raise ValueError(f"K phải từ 2 đến số màu khác nhau sau thu nhỏ ({unique_colors}).")

    figures = ROOT / "docs" / "images" / "image_segmentation"
    reports = ROOT / "reports" / "image_segmentation"
    figures.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    figure, grid = plt.subplots(1, len(cluster_counts) + 1, squeeze=False,
                                figsize=(3.2 * (len(cluster_counts) + 1), 4.5), layout="constrained")
    axes = grid[0]
    axes[0].imshow(original)
    axes[0].set_title(f"Ảnh gốc\n{original.width} × {original.height}")
    records = []
    with threadpool_limits(limits=1):
        for ax, k in zip(axes[1:], cluster_counts, strict=True):
            model = KMeans(n_clusters=k, n_init=10, random_state=42).fit(pixels)
            palette = np.clip(np.rint(model.cluster_centers_), 0, 255).astype(np.uint8)
            rendered_pixels = palette[model.labels_]
            small = Image.fromarray(rendered_pixels.reshape(height, width, 3))
            output = small.resize(original.size, Image.Resampling.NEAREST)
            output.save(figures / f"k_{k}.png")
            mse = float(np.square(pixels - rendered_pixels).mean(dtype=np.float64))
            records.append({
                "image": source.name, "k": k, "working_width": width, "working_height": height,
                "output_width": original.width, "output_height": original.height,
                "mse_rgb": mse, "inertia": float(model.inertia_),
                "palette": " ".join("#{:02x}{:02x}{:02x}".format(*rgb) for rgb in palette),
            })
            ax.imshow(output)
            ax.set_title(f"K = {k}\nMSE RGB {mse:.1f}")
    for ax in axes:
        ax.set_axis_off()
    save(figure, figures / "comparison.png")
    result = pd.DataFrame(records)
    result.to_csv(reports / "metrics.csv", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Phân đoạn ảnh theo màu bằng K-means.")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--k", type=int, nargs="+", default=[3, 6, 9])
    parser.add_argument("--max-side", type=int, default=320)
    args = parser.parse_args()
    print(segment_image(args.image, args.k, args.max_side).to_string(index=False))


if __name__ == "__main__":
    main()
