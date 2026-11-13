import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .model import load_model
from .preprocess import AGE, INCOME, ROOT, SPEND
from .tracking import latest_run


def assign_customer(bundle: dict, age: float, income: float, spending: float) -> dict:
    values = {AGE: age, INCOME: income, SPEND: spending}
    if not np.isfinite(list(values.values())).all():
        raise ValueError("Tuổi, thu nhập và điểm chi tiêu phải là số hữu hạn.")
    if age <= 0 or income < 0 or not 1 <= spending <= 100:
        raise ValueError("Tuổi phải dương, thu nhập từ 0 và điểm chi tiêu trong khoảng 1–100.")
    row = pd.DataFrame([values], columns=bundle["features"])
    return {
        "cluster": int(bundle["model"].predict(row)[0]),
        "outside_train_range": [
            name for name, (low, high) in bundle["ranges"].items()
            if not low <= values[name] <= high
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Gán cụm cho một khách hàng mới.")
    parser.add_argument("--age", type=float, required=True)
    parser.add_argument("--income", type=float, required=True, help="Thu nhập hằng năm, đơn vị nghìn USD.")
    parser.add_argument("--spending", type=float, required=True, help="Điểm chi tiêu từ 1 đến 100.")
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()
    try:
        path = args.model or ROOT / latest_run()["paths"]["model"] / "customer_clusters.joblib"
        result = assign_customer(load_model(path), args.age, args.income, args.spending)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
