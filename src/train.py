"""Thử K trên train/validation, đánh giá test và lưu kết quả thực nghiệm."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate import silhouette_score
from src.model import KMeansModel, load_config, load_model, save_model
from src.preprocessing import check_overlap
from src.tracking import create_run, update_progress, log_experiment
from src.visualize import plot_clusters_3d, plot_model_selection


def read_scaled(root, data_config, name, scaler):
    """Đọc tập đã chuẩn hóa, kiểm tra mã và giá trị khớp dữ liệu gốc cùng scaler train."""
    features = data_config["features"]
    scaled = pd.read_csv(root / data_config[f"{name}_path"])
    original = pd.read_csv(root / "data" / "processed" / f"{name}.csv")
    if not scaled.CustomerID.equals(original.CustomerID):
        raise ValueError(f"Thứ tự CustomerID của {name} không khớp dữ liệu gốc.")
    expected = (original[features] - pd.Series(scaler["mean"])) / pd.Series(scaler["std"])
    if not np.allclose(scaled[features].to_numpy(), expected.to_numpy()):
        raise ValueError(f"Dữ liệu {name} không khớp thông số chuẩn hóa train.")
    return original, scaled[features].to_numpy(dtype=float)


def _run_experiment(root, run_id):
    """Huấn luyện, chọn K, đánh giá test và ghi kết quả cho lần chạy được chỉ định."""
    root = Path(root)
    config = load_config(root / "config.yaml")
    if config["experiment"]["selection_metric"] != "validation_silhouette":
        raise ValueError("Chỉ hỗ trợ chọn K theo validation_silhouette.")
    data_config = config["data"]
    scaler = json.loads((root / data_config["scaler_path"]).read_text(encoding="utf-8"))
    if scaler["fit_scope"] != "train_only" or scaler["features"] != data_config["features"]:
        raise ValueError("Scaler phải học trên train và có cùng thứ tự đặc trưng cấu hình.")
    train_raw, train = read_scaled(root, data_config, "train", scaler)
    val_raw, validation = read_scaled(root, data_config, "validation", scaler)
    if set(train_raw.CustomerID) & set(val_raw.CustomerID):
        raise ValueError("Train và validation bị chồng lấn CustomerID.")
    train_features = set(map(tuple, train_raw[data_config["features"]].to_numpy()))
    if train_features & set(map(tuple, val_raw[data_config["features"]].to_numpy())):
        raise ValueError("Train và validation bị chồng lấn đặc trưng; cần xử lý trước.")
    results = []
    models = {}
    update_progress(root, run_id, "MODEL_SEARCH", "STARTED")
    for k in config["experiment"]["cluster_counts"]:
        params = {**config["model"], "n_clusters": k}
        model = KMeansModel(**params).build_model().fit(train)
        labels = model.predict(validation)
        n_present = int(len(np.unique(labels)))
        # Chỉ so sánh K nếu cả K cụm đều xuất hiện trên validation.
        valid = n_present == k and 2 <= n_present < len(validation)
        score = silhouette_score(validation, labels) if valid else np.nan
        results.append({"k": k, "train_inertia": model.inertia_, "validation_silhouette": score,
                        "validation_clusters_present": n_present, "iterations": model.n_iter_,
                        "converged": model.converged_})
        models[k] = model
    table = pd.DataFrame(results)
    candidates = table.dropna(subset=["validation_silhouette"])
    if candidates.empty:
        raise ValueError("Không có K hợp lệ để tính Silhouette trên validation.")
    # Nếu điểm bằng nhau, chọn K nhỏ hơn theo thứ tự tăng dần.
    chosen = int(candidates.sort_values(["validation_silhouette", "k"], ascending=[False, True]).iloc[0]["k"])
    model = models[chosen]
    update_progress(root, run_id, "MODEL_SEARCH", "SUCCESS")
    update_progress(root, run_id, "TEST_EVALUATION", "STARTED")
    # Chỉ đọc và đánh giá test sau khi lựa chọn K hoàn tất.
    test_raw, test = read_scaled(root, data_config, "test", scaler)
    if (set(test_raw.CustomerID) & (set(train_raw.CustomerID) | set(val_raw.CustomerID))):
        raise ValueError("Test bị chồng lấn CustomerID với train hoặc validation.")
    overlap, _ = check_overlap({"train": train_raw, "validation": val_raw, "test": test_raw})
    if overlap.shared_feature_combinations.any():
        raise ValueError("Các tập bị chồng lấn đặc trưng; cần chuyển về train trước.")
    test_labels = model.predict(test)
    n_test_clusters = int(len(np.unique(test_labels)))
    test_score = silhouette_score(test, test_labels) if 2 <= n_test_clusters < len(test) else None
    update_progress(root, run_id, "TEST_EVALUATION", "SUCCESS")
    train_raw["cluster"] = model.labels_
    val_raw["cluster"] = model.predict(validation)
    test_raw["cluster"] = test_labels
    profile = train_raw.groupby("cluster")[data_config["features"]].mean()
    profile.insert(0, "count", train_raw.groupby("cluster").size())
    metrics = {"chosen_k": chosen, "selection": "highest_valid_validation_silhouette",
               "train_rows": len(train), "validation_rows": len(validation), "test_rows": len(test),
               "train_inertia": model.inertia_, "validation_silhouette": float(candidates.loc[candidates.k == chosen, "validation_silhouette"].iloc[0]),
               "test_silhouette": test_score, "test_clusters_present": n_test_clusters,
               "iterations": model.n_iter_, "converged": bool(model.converged_),
               "initial_config": config["model"], "cluster_counts_tried": config["experiment"]["cluster_counts"]}
    reports, images, models_dir = root / "reports", root / "docs/images", root / "models"
    for path in (reports, images, models_dir):
        path.mkdir(parents=True, exist_ok=True)
    update_progress(root, run_id, "SAVE_RESULTS", "STARTED")
    table.to_csv(reports / "k_selection.csv", index=False)
    for name, frame in (("train", train_raw), ("validation", val_raw), ("test", test_raw)):
        frame.to_csv(reports / f"{name}_clusters.csv", index=False)
    profile.to_csv(reports / "cluster_profile.csv")
    (reports / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    save_model(model, scaler, models_dir / "kmeans.json")
    restored, restored_scaler = load_model(models_dir / "kmeans.json")
    if restored_scaler != scaler or not np.array_equal(restored.predict(test), test_labels):
        raise AssertionError("Mô hình nạp lại không tái tạo kết quả dự đoán.")
    plot_model_selection(table, chosen, images / "elbow_silhouette.png")
    plot_clusters_3d(train_raw, model.labels_, images / "train_clusters_3d.png")
    log_experiment(root, run_id, config, metrics)
    update_progress(root, run_id, "SAVE_RESULTS", "SUCCESS")
    return table, metrics, profile


def run_experiment(root):
    """Tạo log, chạy thực nghiệm, ghi trạng thái thành công hoặc lỗi rồi trả kết quả."""
    root = Path(root)
    run_id = create_run(root)
    try:
        result = _run_experiment(root, run_id)
        update_progress(root, run_id, "TRAINING", "SUCCESS")
        print(f"Lần chạy: {run_id}; log: logs/{run_id}.log")
        return result
    except Exception as error:
        update_progress(root, run_id, "TRAINING", "FAILED", str(error))
        raise


if __name__ == "__main__":
    table, metrics, profile = run_experiment(Path(__file__).resolve().parents[1])
    print(table.to_string(index=False))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(profile.to_string())
