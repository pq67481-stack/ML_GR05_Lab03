"""Ghi trạng thái và kết quả từng lần huấn luyện theo cách đơn giản."""

import csv
import hashlib
import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd


def create_run(root):
    """Tạo mã lần chạy duy nhất và thư mục lưu log, trạng thái, thực nghiệm."""
    root = Path(root)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    for name in ("logs", "progress_results", "reports/experiments"):
        (root / name).mkdir(parents=True, exist_ok=True)
    update_progress(root, run_id, "TRAINING", "STARTED")
    return run_id


def update_progress(root, run_id, step, status, error=None):
    """Ghi trạng thái mới nhất vào JSON và nối sự kiện có thời gian UTC vào log."""
    root = Path(root)
    record = {"run_id": run_id, "step": step, "status": status,
              "timestamp_utc": datetime.now(timezone.utc).isoformat(), "error": error}
    (root / "progress_results" / f"{run_id}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    with (root / "logs" / f"{run_id}.log").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def file_hash(path):
    """Tính SHA-256 của tệp để kiểm tra đầu vào của lần chạy."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def log_experiment(root, run_id, config, metrics):
    """Lưu cấu hình, kết quả, model và dấu vết dữ liệu riêng cho lần chạy thành công."""
    root = Path(root)
    folder = root / "reports" / "experiments" / run_id
    folder.mkdir(parents=True, exist_ok=False)
    inputs = {key: {"path": value, "sha256": file_hash(root / value)}
              for key, value in config["data"].items() if key.endswith("_path")}
    record = {"run_id": run_id, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
              "config": config, "metrics": metrics, "inputs": inputs,
              "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}}
    (folder / "experiment.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(root / "models/kmeans.json", folder / "kmeans.json")
    shutil.copy2(root / "reports/k_selection.csv", folder / "k_selection.csv")
    summary = root / "reports/experiment_logs.csv"
    fields = ["run_id", "finished_at_utc", "chosen_k", "train_inertia", "validation_silhouette", "test_silhouette"]
    is_new = not summary.exists() or summary.stat().st_size == 0
    with summary.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if is_new:
            writer.writeheader()
        writer.writerow({"run_id": run_id, "finished_at_utc": record["finished_at_utc"],
                         **{key: metrics[key] for key in fields[2:]}})
    return folder
