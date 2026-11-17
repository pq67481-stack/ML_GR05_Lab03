"""Đối chiếu độc lập dữ liệu ứng viên; không ghi đè kết quả bài lab."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate import silhouette_score
from src.model import KMeansModel
from src.visualize import plot_candidate_review


def examine(path):
    """Kiểm tra dữ liệu và thử chia nhiều seed, chọn K bằng validation."""
    data = pd.read_csv(path)
    columns = ["age", "income", "spending_score"]
    X = data[columns].to_numpy(dtype=float)
    results = []
    for seed in (0, 1, 2, 3, 42):
        order = np.random.default_rng(seed).permutation(len(X))
        n_train, n_val = int(len(X) * 0.7), int(len(X) * 0.15)
        train, val, test = X[order[:n_train]], X[order[n_train:n_train+n_val]], X[order[n_train+n_val:]]
        mean, std = train.mean(axis=0), train.std(axis=0)
        train, val, test = (train - mean)/std, (val - mean)/std, (test - mean)/std
        options = []
        for k in range(2, 9):
            model = KMeansModel(n_clusters=k, n_init=5, max_iter=100, random_state=42).build_model().fit(train)
            labels = model.predict(val)
            if len(np.unique(labels)) == k:
                options.append((silhouette_score(val, labels), -k, model))
        score, neg_k, model = max(options, key=lambda item: (item[0], item[1]))
        test_labels = model.predict(test)
        results.append({"split_seed": seed, "k": -neg_k, "validation_silhouette": score,
                        "test_silhouette": silhouette_score(test, test_labels),
                        "test_clusters": len(np.unique(test_labels))})
    report = {"rows": len(data), "columns": list(data.columns), "missing": data.isna().sum().to_dict(),
              "duplicate_rows": int(data.duplicated().sum()), "duplicate_ids": int(data.id.duplicated().sum()),
              "duplicate_features": int(data.duplicated(columns).sum()),
              "summary": data[columns].describe().to_dict(), "correlations": data[columns].corr().to_dict(),
              "gender_counts": data.gender.value_counts().to_dict(), "splits": results}
    return report


if __name__ == "__main__":
    report = examine(Path(__file__).with_name("candidate_1000.csv"))
    Path(__file__).with_name("candidate_review.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_candidate_review(pd.read_csv(Path(__file__).with_name("candidate_1000.csv")),
                          pd.DataFrame(report["splits"]),
                          Path(__file__).resolve().parents[1] / "docs/images/candidate_review.png")
    print(json.dumps(report, ensure_ascii=False, indent=2))
