"""Đọc, chuẩn bị và phân vùng ảnh bằng K-means của bài chính."""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from src.model import KMeansModel


def load_config(path):
    """Đọc cấu hình JSON hợp lệ trong YAML và trả về dictionary."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_image(path):
    """Đọc ảnh, áp dụng hướng EXIF và chuyển về RGB, giữ kích thước sau xoay."""
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def prepare_image(image, max_side):
    """Thu nhỏ ảnh giữ tỷ lệ và trả ảnh nhỏ cùng ma trận RGB của pixel."""
    if not isinstance(max_side, int) or max_side < 1:
        raise ValueError("train_max_side phải là số nguyên dương.")
    small = image.copy()
    small.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    pixels = np.asarray(small, dtype=np.float64).reshape(-1, 3)
    return small, pixels


def segment_image(image, train_pixels, k, config):
    """Huấn luyện tâm RGB trên ảnh nhỏ rồi gán toàn bộ ảnh gốc theo từng lô."""
    model = KMeansModel(n_clusters=k, n_init=config["n_init"], max_iter=config["max_iter"],
                        tol=config["tol"], random_state=config["random_state"]).build_model()
    model.fit(train_pixels)
    pixels = np.asarray(image, dtype=np.float64).reshape(-1, 3)
    batch_size = config["prediction_batch_size"]
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("prediction_batch_size phải là số nguyên dương.")
    labels = np.concatenate([model.predict(pixels[start:start + batch_size])
                             for start in range(0, len(pixels), batch_size)])
    colors = np.rint(model.cluster_centers_[labels]).clip(0, 255).astype(np.uint8)
    segmented = colors.reshape(image.height, image.width, 3)
    return model, labels, segmented
