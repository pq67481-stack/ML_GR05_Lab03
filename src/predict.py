"""Nạp mô hình và chuyển dữ liệu khách hàng gốc thành nhãn cụm."""

import argparse
from pathlib import Path

import numpy as np
from src.model import load_model


def predict_customer(age, annual_income, spending_score, model_path):
    """Chuẩn hóa ba giá trị theo scaler đã lưu, trả nhãn và tâm ở đơn vị gốc."""
    values = np.array([age, annual_income, spending_score], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Các giá trị phải là số hữu hạn.")
    if values[0] <= 0 or values[1] < 0 or not 1 <= values[2] <= 100:
        raise ValueError("Tuổi phải lớn hơn 0, thu nhập không âm, điểm chi tiêu từ 1 đến 100.")
    model, scaler = load_model(Path(model_path))
    features = ["Age", "Annual Income", "Spending Score"]
    if scaler["features"] != features or scaler["fit_scope"] != "train_only":
        raise ValueError("Thông số chuẩn hóa không phù hợp ba đặc trưng của bài lab.")
    mean = np.array([scaler["mean"][name] for name in features], dtype=float)
    std = np.array([scaler["std"][name] for name in features], dtype=float)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Thông số chuẩn hóa đã lưu không hợp lệ.")
    standardized = (values - mean) / std
    cluster = int(model.predict(standardized.reshape(1, -1))[0])
    center = model.cluster_centers_[cluster] * std + mean
    return {"cluster": cluster, "center": dict(zip(features, center.tolist()))}


def main():
    """Nhận ba đặc trưng từ terminal và in cụm dự đoán cùng tâm cụm."""
    parser = argparse.ArgumentParser(description="Gán cụm khách hàng bằng K-means đã huấn luyện.")
    parser.add_argument("--age", type=float, required=True, help="Tuổi khách hàng")
    parser.add_argument("--income", type=float, required=True, help="Thu nhập năm, cùng đơn vị dữ liệu gốc")
    parser.add_argument("--spending", type=float, required=True, help="Điểm chi tiêu 1–100")
    parser.add_argument("--model", default=str(Path(__file__).resolve().parents[1] / "models/kmeans.json"))
    args = parser.parse_args()
    result = predict_customer(args.age, args.income, args.spending, args.model)
    print(f"Cụm dự đoán: {result['cluster']}")
    for feature, value in result["center"].items():
        print(f"Tâm cụm — {feature}: {value:.2f}")


if __name__ == "__main__":
    main()
