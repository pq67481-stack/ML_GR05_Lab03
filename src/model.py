"""K-means tự triển khai bằng NumPy, theo thứ tự xây dựng của Lab 01."""

import json
from pathlib import Path

import numpy as np


def load_config(path):
    """Đọc cấu hình định dạng JSON hợp lệ trong YAML; trả dict hoặc báo lỗi."""
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def _as_matrix(X):
    """Chuyển X thành ma trận số 2 chiều hữu hạn, ít nhất một mẫu và đặc trưng."""
    matrix = np.asarray(X, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("X phải là ma trận hai chiều có ít nhất một mẫu và một đặc trưng.")
    if not np.isfinite(matrix).all():
        raise ValueError("X không được chứa giá trị thiếu hoặc vô hạn.")
    return matrix


def squared_distances(X, centers):
    """Tính ma trận bình phương khoảng cách Euclid giữa mẫu X và từng tâm."""
    # Mỗi phần tử là tổng (x_j - tâm_j)² qua tất cả các đặc trưng j.
    return ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)


def assign_clusters(X, centers):
    """Gán mỗi mẫu vào tâm gần nhất và trả nhãn cùng khoảng cách bình phương."""
    distances = squared_distances(X, centers)
    labels = distances.argmin(axis=1)
    return labels, distances[np.arange(len(X)), labels]


def update_centers(X, labels, distances, n_clusters):
    """Tính trung bình mỗi cụm; xử lý cụm rỗng bằng điểm xa tâm từ cụm có dư."""
    labels = labels.copy()
    counts = np.bincount(labels, minlength=n_clusters)
    for empty in np.flatnonzero(counts == 0):
        eligible = counts[labels] > 1
        # Chuyển điểm xa nhất, nhưng không lấy điểm cuối cùng của cụm nguồn.
        index = int(np.argmax(np.where(eligible, distances, -np.inf)))
        source = labels[index]
        labels[index] = empty
        counts[source] -= 1
        counts[empty] += 1
    centers = np.vstack([X[labels == cluster].mean(axis=0) for cluster in range(n_clusters)])
    return centers


def save_model(model, scaler, path):
    """Lưu mô hình đã fit, cấu hình và thông số chuẩn hóa train vào JSON."""
    if not hasattr(model, "cluster_centers_"):
        raise ValueError("Cần fit trước khi lưu mô hình.")
    payload = {"model": {key: getattr(model, key) for key in
                         ("n_clusters", "init", "n_init", "max_iter", "tol", "random_state")},
               "cluster_centers": model.cluster_centers_.tolist(), "inertia": model.inertia_,
               "n_iter": model.n_iter_, "converged": model.converged_, "scaler": scaler}
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_model(path):
    """Nạp mô hình và thông số chuẩn hóa từ JSON, trả về cặp (model, scaler)."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    model = KMeansModel(**payload["model"]).build_model()
    model.cluster_centers_ = _as_matrix(payload["cluster_centers"])
    if len(model.cluster_centers_) != model.n_clusters:
        raise ValueError("Số tâm đã lưu không khớp cấu hình.")
    model.inertia_ = payload["inertia"]
    model.n_iter_ = payload["n_iter"]
    model.converged_ = payload["converged"]
    return model, payload["scaler"]


class KMeansModel:
    """Phân cụm K-means: khởi tạo, build_model, fit rồi predict; X không có mã khách."""

    def __init__(self, n_clusters=3, init="random", n_init=5, max_iter=100, tol=0.0001, random_state=42):
        """Ghi siêu tham số; đối tượng chưa được xây dựng và chưa có tâm cụm."""
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self._built = False

    def build_model(self):
        """Kiểm tra siêu tham số và xóa trạng thái học cũ để chuẩn bị fit."""
        for name in ("n_clusters", "n_init", "max_iter"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} phải là số nguyên dương.")
        if self.init != "random":
            raise ValueError("Hiện chỉ hỗ trợ init='random'.")
        if isinstance(self.tol, bool) or not isinstance(self.tol, (int, float)) or not np.isfinite(self.tol) or self.tol < 0:
            raise ValueError("tol phải là số hữu hạn không âm.")
        if isinstance(self.random_state, bool) or not isinstance(self.random_state, (int, np.integer)) or self.random_state < 0:
            raise ValueError("random_state phải là số nguyên không âm.")
        for name in ("cluster_centers_", "labels_", "inertia_", "n_iter_", "converged_"):
            if hasattr(self, name):
                delattr(self, name)
        self._built = True
        return self

    def fit(self, X):
        """Huấn luyện từ ma trận số X; thử nhiều tâm ban đầu và giữ inertia thấp nhất."""
        if not self._built:
            raise ValueError("Cần gọi build_model() trước khi fit.")
        # Không giữ nghiệm cũ sau lần fit lỗi hoặc chạy lại.
        for name in ("cluster_centers_", "labels_", "inertia_", "n_iter_", "converged_"):
            if hasattr(self, name):
                delattr(self, name)
        X = _as_matrix(X)
        unique = np.unique(X, axis=0)
        if len(unique) < self.n_clusters:
            raise ValueError("Số điểm khác nhau phải lớn hơn hoặc bằng n_clusters.")
        rng = np.random.default_rng(self.random_state)
        best = None
        for _ in range(self.n_init):
            # Chọn K điểm khác giá trị làm tâm ban đầu.
            centers = unique[rng.choice(len(unique), size=self.n_clusters, replace=False)].copy()
            converged = False
            for iteration in range(1, self.max_iter + 1):
                labels, distances = assign_clusters(X, centers)
                updated = update_centers(X, labels, distances, self.n_clusters)
                movement = np.linalg.norm(updated - centers)
                centers = updated
                if movement <= self.tol:
                    converged = True
                    break
            labels, distances = assign_clusters(X, centers)
            inertia = float(distances.sum())
            if best is None or inertia < best[0]:
                best = (inertia, centers.copy(), labels.copy(), iteration, converged)
        self.inertia_, self.cluster_centers_, self.labels_, self.n_iter_, self.converged_ = best
        return self

    def predict(self, X):
        """Gán mẫu mới đã chuẩn hóa vào tâm gần nhất, không cập nhật tâm."""
        if not hasattr(self, "cluster_centers_"):
            raise ValueError("Cần fit mô hình trước khi predict.")
        X = _as_matrix(X)
        if X.shape[1] != self.cluster_centers_.shape[1]:
            raise ValueError("Số đặc trưng không khớp dữ liệu huấn luyện.")
        return assign_clusters(X, self.cluster_centers_)[0]
