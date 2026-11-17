"""Tự tính Silhouette bằng NumPy trên các đặc trưng đã chuẩn hóa."""

import numpy as np


def silhouette_samples(X, labels, batch_size=128):
    """Tính Silhouette chính xác theo từng lô để giảm bộ nhớ; cụm một điểm nhận 0."""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    if X.ndim != 2 or X.shape[1] == 0 or not np.isfinite(X).all():
        raise ValueError("X phải là ma trận số hữu hạn hai chiều.")
    if labels.ndim != 1 or len(labels) != len(X):
        raise ValueError("labels phải có một nhãn cho mỗi mẫu.")
    clusters = np.unique(labels)
    if not 2 <= len(clusters) < len(X):
        raise ValueError("Silhouette cần từ 2 đến số mẫu trừ 1 cụm có mặt.")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size phải là số nguyên dương.")
    result = np.zeros(len(X))
    for start in range(0, len(X), batch_size):
        stop = min(start + batch_size, len(X))
        distances = np.sqrt(((X[start:stop, None, :] - X[None, :, :]) ** 2).sum(axis=2))
        # Chia lô theo điểm được xét; vẫn so với toàn bộ điểm trong tập đánh giá.
        for offset, label in enumerate(labels[start:stop]):
            same = labels == label
            if same.sum() == 1:
                continue
            a = distances[offset, same].sum() / (same.sum() - 1)
            b = min(distances[offset, labels == other].mean() for other in clusters if other != label)
            if max(a, b) > 0:
                result[start + offset] = (b - a) / max(a, b)
    return result


def silhouette_score(X, labels):
    """Trả trung bình Silhouette của các điểm trong cùng một tập dữ liệu."""
    return float(silhouette_samples(X, labels).mean())
