"""Đọc dữ liệu khách hàng và chuẩn bị đặc trưng cho phân cụm."""

from pathlib import Path
import json

import numpy as np
import pandas as pd

FEATURES = ["Age", "Annual Income", "Spending Score"]
REQUIRED = ["CustomerID", "Gender", *FEATURES]


def load_data(path):
    """Đọc CSV gốc, kiểm tra các cột bắt buộc và trả về bảng dữ liệu."""
    data = pd.read_csv(path)
    data = data.rename(columns={"Customer ID": "CustomerID"})
    missing = [name for name in REQUIRED if name not in data.columns]
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {missing}")
    return data


def inspect_data(data):
    """Trả về bảng chất lượng và các dòng có đặc trưng nằm ngoài ngưỡng IQR."""
    quality = pd.DataFrame({"dtype": data.dtypes.astype(str), "missing": data.isna().sum(),
                            "unique": data.nunique()})
    numeric = data[FEATURES].apply(pd.to_numeric, errors="raise")
    q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
    iqr = q3 - q1
    flags = (numeric.lt(q1 - 1.5 * iqr) | numeric.gt(q3 + 1.5 * iqr))
    flagged = pd.concat([data[["CustomerID"]], numeric, flags.add_suffix("_flag")], axis=1)
    return quality, flagged.loc[flags.any(axis=1)], flags.sum()


def prepare_features(data):
    """Chọn ba đặc trưng và chuẩn hóa Z-score; trả bảng, trung bình và độ lệch chuẩn."""
    if data["CustomerID"].isna().any() or data["CustomerID"].duplicated().any():
        raise ValueError("CustomerID phải đầy đủ và không trùng để truy vết.")
    values = data[FEATURES].apply(pd.to_numeric, errors="raise").astype(float)
    if values.isna().any().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError("Đặc trưng chứa dữ liệu thiếu hoặc không hữu hạn; cần xử lý trước.")
    mean = values.mean()
    std = values.std(ddof=0)
    if (std == 0).any():
        raise ValueError("Không thể chuẩn hóa cột có độ lệch chuẩn bằng 0.")
    scaled = (values - mean) / std
    scaled.insert(0, "CustomerID", data["CustomerID"].to_numpy())
    return scaled, mean, std


def save_prepared(data, scaled, mean, std, output_dir):
    """Lưu dữ liệu chọn đặc trưng, dữ liệu chuẩn hóa và thông số để tái lập."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data[["CustomerID", *FEATURES]].to_csv(output / "features.csv", index=False)
    scaled.to_csv(output / "features_scaled.csv", index=False)
    params = {"method": "z_score", "ddof": 0, "fit_scope": "all_rows_descriptive",
              "features": FEATURES, "mean": mean.to_dict(), "std": std.to_dict()}
    (output / "scaler.json").write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")


def clean_data(data):
    """Làm sạch dòng trùng hoàn toàn, kiểm tra dữ liệu; trả bảng sạch và số liệu xử lý.

    Bỏ khoảng trắng ở giới tính, không tự xóa ngoại lệ hay điền dữ liệu thiếu.
    Các mã trùng, giá trị không hợp lệ hoặc thiếu làm hàm dừng để xem xét.
    """
    clean = data.copy()
    clean["Gender"] = clean["Gender"].str.strip()
    duplicates = int(clean.duplicated().sum())
    clean = clean.drop_duplicates().reset_index(drop=True)
    if clean[REQUIRED].isna().any().any():
        raise ValueError("Có dữ liệu bắt buộc bị thiếu; cần quyết định cách xử lý.")
    if clean["CustomerID"].duplicated().any():
        raise ValueError("Có CustomerID trùng nhưng nội dung khác nhau; cần kiểm tra.")
    if not clean["Gender"].isin(["Male", "Female"]).all():
        raise ValueError("Có giá trị Gender không hợp lệ; cần kiểm tra.")
    values = clean[FEATURES].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Có đặc trưng không hữu hạn.")
    if (values["Age"] <= 0).any() or (values["Annual Income"] < 0).any():
        raise ValueError("Có tuổi hoặc thu nhập không hợp lệ.")
    if not values["Spending Score"].between(1, 100).all():
        raise ValueError("Điểm chi tiêu phải nằm trong khoảng 1–100.")
    clean[FEATURES] = values
    return clean, {"input_rows": len(data), "duplicate_rows_removed": duplicates,
                   "clean_rows": len(clean)}


def split_data(data, random_state=42):
    """Chia ngẫu nhiên 70% train, 15% validation, phần còn lại test bằng NumPy."""
    order = np.random.default_rng(random_state).permutation(len(data))
    n_train = int(len(data) * 0.70)
    n_val = int(len(data) * 0.15)
    return {"train": data.iloc[order[:n_train]].reset_index(drop=True),
            "validation": data.iloc[order[n_train:n_train + n_val]].reset_index(drop=True),
            "test": data.iloc[order[n_train + n_val:]].reset_index(drop=True)}


def check_overlap(splits):
    """Đếm chồng lấn mã khách và tổ hợp ba đặc trưng giữa từng cặp tập.

    Tổ hợp đặc trưng giống nhau có thể thuộc khách khác nhau và không tự chứng minh rò rỉ.
    """
    rows = []
    details = []
    pairs = [("train", "validation"), ("train", "test"), ("validation", "test")]
    for left, right in pairs:
        a, b = splits[left], splits[right]
        common_ids = set(a.CustomerID) & set(b.CustomerID)
        shared = a[["CustomerID", *FEATURES]].merge(
            b[["CustomerID", *FEATURES]], on=FEATURES, suffixes=("_left", "_right"))
        rows.append({"split_a": left, "split_b": right, "shared_customer_ids": len(common_ids),
                     "shared_feature_combinations": len(shared[FEATURES].drop_duplicates()),
                     "matching_customer_pairs": len(shared)})
        if not shared.empty:
            shared.insert(0, "split_b", right)
            shared.insert(0, "split_a", left)
            details.append(shared)
    columns = ["split_a", "split_b", "CustomerID_left", *FEATURES, "CustomerID_right"]
    return pd.DataFrame(rows), (pd.concat(details, ignore_index=True) if details else pd.DataFrame(columns=columns))


def resolve_overlap(splits):
    """Chuyển mọi khách có mã hoặc bộ ba đặc trưng chung giữa các tập về train.

    Khử dòng giống hệt sau khi chuyển; nếu cùng mã có nội dung mâu thuẫn,
    clean_data sẽ báo lỗi thay vì tự chọn bản ghi. Trả tập mới, chi tiết chuyển
    và số dòng trùng hoàn toàn được xóa trong train sau khi chuyển.
    """
    names = ("train", "validation", "test")
    result = {name: splits[name].copy() for name in names}
    shared_ids, shared_features = set(), set()
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            shared_ids |= set(result[left]["CustomerID"]) & set(result[right]["CustomerID"])
            a = set(map(tuple, result[left][FEATURES].to_numpy()))
            b = set(map(tuple, result[right][FEATURES].to_numpy()))
            shared_features |= a & b
    moved = []
    for name in ("validation", "test"):
        frame = result[name]
        by_id = frame["CustomerID"].isin(shared_ids)
        by_features = pd.Series([tuple(row) in shared_features for row in frame[FEATURES].to_numpy()], index=frame.index)
        mask = by_id | by_features
        selected = frame.loc[mask].copy()
        if not selected.empty:
            selected.insert(0, "source_split", name)
            selected.insert(1, "overlap_id", by_id[mask].to_numpy())
            selected.insert(2, "overlap_features", by_features[mask].to_numpy())
            moved.append(selected)
            result["train"] = pd.concat([result["train"], frame.loc[mask]], ignore_index=True)
        result[name] = frame.loc[~mask].reset_index(drop=True)
    result["train"], cleaning = clean_data(result["train"])
    columns = ["source_split", "overlap_id", "overlap_features", *splits["train"].columns]
    details = pd.concat(moved, ignore_index=True) if moved else pd.DataFrame(columns=columns)
    return result, details, cleaning["duplicate_rows_removed"]


def scale_splits(splits):
    """Học trung bình và độ lệch chuẩn từ train rồi áp dụng cho cả ba tập."""
    train_scaled, mean, std = prepare_features(splits["train"])
    scaled = {"train": train_scaled}
    for name in ("validation", "test"):
        frame = splits[name]
        values = frame[FEATURES].astype(float)
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError(f"Tập {name} có giá trị không hữu hạn.")
        result = (values - mean) / std
        result.insert(0, "CustomerID", frame.CustomerID.to_numpy())
        scaled[name] = result
    return scaled, mean, std


def save_splits(clean, splits, scaled, mean, std, output_dir, random_state=42):
    """Lưu dữ liệu sạch, từng tập trước/sau chuẩn hóa và thông số học từ train."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output / "cleaned.csv", index=False)
    for name in splits:
        splits[name].to_csv(output / f"{name}.csv", index=False)
        scaled[name].to_csv(output / f"{name}_scaled.csv", index=False)
    params = {"method": "z_score", "ddof": 0, "fit_scope": "train_only",
              "features": FEATURES, "random_state": random_state,
              "mean": mean.to_dict(), "std": std.to_dict()}
    (output / "train_scaler.json").write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
