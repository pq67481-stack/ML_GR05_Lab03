from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
ID = "CustomerID"
GENDER = "Gender"
AGE = "Age"
INCOME = "Annual Income (k$)"
SPEND = "Spending Score (1-100)"


def load_config(path: Path = ROOT / "config.yaml") -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def clean_customers(raw: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    expected = [ID, GENDER, *features]
    missing = set(expected) - set(raw.columns)
    if missing:
        raise ValueError(f"Thiếu cột dữ liệu: {sorted(missing)}")
    clean = raw.loc[:, expected].drop_duplicates().copy()
    if clean[ID].isna().any() or clean[ID].duplicated().any():
        raise ValueError("CustomerID phải có giá trị và duy nhất sau khi bỏ dòng trùng.")
    for feature in features:
        clean[feature] = pd.to_numeric(clean[feature], errors="raise")
    if np.isinf(clean[features].to_numpy(dtype=float)).any():
        raise ValueError("Các biến số phải hữu hạn.")
    return clean.sort_values(ID).reset_index(drop=True)


def make_preprocessor() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])
