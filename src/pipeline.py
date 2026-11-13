import argparse
import hashlib
import io
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from .eda import summarize
from .evaluate import check_stability, evaluate, profile
from .model import save_model
from .preprocess import AGE, ID, INCOME, ROOT, SPEND, clean_customers, load_config
from .report import write_report
from .tracking import Run, write_json
from .train import fit_selected, search
from .visualize import plot_eda, plot_search, plot_segments, plot_silhouette


def _evaluate_partition(frame: pd.DataFrame, bundle: dict, output: Path) -> tuple[dict, pd.DataFrame]:
    features = bundle["features"]
    model = bundle["model"]
    standardized = model["preprocess"].transform(frame[features])
    labels = model.predict(frame[features])
    metrics, samples = evaluate(standardized, labels)
    z_scores = pd.DataFrame(standardized, index=frame.index, columns=[f"z_{name}" for name in features])
    result = frame.join(z_scores).assign(cluster=labels, silhouette=samples)
    result.to_csv(output, index=False, encoding="utf-8")
    return metrics, result


def run_pipeline(config_path: Path = ROOT / "config.yaml") -> dict:
    run = Run()
    try:
        run.update("data", "Đọc CSV, kiểm tra dữ liệu và chia train/test.")
        config = load_config(config_path)
        features = config["features"]
        if len(features) != 3 or set(features) != {AGE, INCOME, SPEND}:
            raise ValueError("Lab yêu cầu ba biến số Age, Annual Income và Spending Score.")
        raw_path = ROOT / config["data"]["path"]
        raw_bytes = raw_path.read_bytes()
        raw = pd.read_csv(io.BytesIO(raw_bytes))
        clean = clean_customers(raw, features)
        train, test = train_test_split(
            clean, test_size=config["data"]["test_size"], random_state=config["data"]["seed"]
        )
        if train[features].notna().sum().eq(0).any():
            raise ValueError("Mỗi biến đầu vào phải có quan sát hợp lệ trong tập train.")
        train, test = train.sort_values(ID), test.sort_values(ID)
        reports, images = run.paths["reports"], run.paths["images"]
        audit = summarize(raw, clean, features)
        write_json(reports / "eda.json", audit)
        write_json(reports / "experiment.json", {
            "config": config,
            "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                         "pandas": pd.__version__, "numpy": np.__version__},
        })
        plot_eda(clean, features, images)

        run.update("cv", "So sánh K-means và Gaussian Mixture trên các fold của train.")
        with threadpool_limits(limits=1):
            scores, folds = search(train, config)
            scores.to_csv(reports / "cv_summary.csv", index=False)
            folds.to_csv(reports / "cv_folds.csv", index=False)
            bundle = fit_selected(train, scores, config)

            run.update("evaluation", f"Đánh giá cấu hình {bundle['family']} k={bundle['k']} đã chọn.")
            train_metrics, train_rows = _evaluate_partition(train, bundle, run.paths["data"] / "train.csv")
            test_metrics, test_rows = _evaluate_partition(test, bundle, run.paths["data"] / "test.csv")
            stability = check_stability(bundle, train, test, config)

        stability.to_csv(reports / "stability.csv", index=False)
        metrics = {
            "family": bundle["family"],
            "k": bundle["k"],
            "train_rows": len(train),
            "test_rows": len(test),
            "train": train_metrics,
            "test": test_metrics,
            "stability": {
                "repeats": len(stability),
                "mean_ari": float(stability["ari"].mean()),
                "std_ari": float(stability["ari"].std(ddof=0)),
                "median_ari": float(stability["ari"].median()),
                "min_ari": float(stability["ari"].min()),
                "max_ari": float(stability["ari"].max()),
            },
        }
        write_json(reports / "metrics.json", metrics)
        train_profile = profile(train_rows, features)
        test_profile = profile(test_rows, features)
        train_profile.to_csv(reports / "train_profile.csv", index=False)
        test_profile.to_csv(reports / "test_profile.csv", index=False)
        plot_search(scores, images)
        plot_segments(train_rows, test_rows, images)
        plot_silhouette(train_rows, test_rows, images)
        save_model(bundle, run.paths["model"] / "customer_clusters.joblib")
        write_report(reports / "results.md", run.id, audit, scores, metrics, train_profile, test_profile)
        return run.update("completed", "Đã lưu mô hình, dữ liệu, biểu đồ và báo cáo.", "completed")
    except Exception as error:
        run.update("failed", str(error), "failed")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy thực nghiệm phân cụm Lab 03.")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = parser.parse_args()
    manifest = run_pipeline(args.config)
    print(f"Run: {manifest['run_id']}")
    print(ROOT / manifest["paths"]["reports"] / "results.md")


if __name__ == "__main__":
    main()
