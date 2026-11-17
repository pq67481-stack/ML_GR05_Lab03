"""Kiểm tra đầu vào demo và cách ghi trạng thái thực nghiệm."""

import json
import tempfile
import unittest
from pathlib import Path

from src.predict import predict_customer
from src.model import load_model
from src.tracking import create_run, update_progress
from src.train import run_experiment


class TestDemoAndTracking(unittest.TestCase):
    """Kiểm chứng dự đoán từ model thật và log khi chạy thành công hoặc lỗi."""

    def test_predict_and_invalid_input(self):
        """Dự đoán đúng cụm ví dụ và từ chối điểm chi tiêu ngoài khoảng."""
        model = Path(__file__).resolve().parents[1] / "models/kmeans.json"
        result = predict_customer(35, 80000, 60, model)
        self.assertIn(result["cluster"], range(load_model(model)[0].n_clusters))
        self.assertEqual(set(result["center"]), {"Age", "Annual Income", "Spending Score"})
        with self.assertRaises(ValueError):
            predict_customer(35, 80000, 101, model)

    def test_tracking_records_steps(self):
        """Tạo run và kiểm tra JSON trạng thái cùng các sự kiện trong log."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run_id = create_run(root)
            update_progress(root, run_id, "MODEL_SEARCH", "SUCCESS")
            state = json.loads((root / "progress_results" / f"{run_id}.json").read_text(encoding="utf-8"))
            lines = (root / "logs" / f"{run_id}.log").read_text(encoding="utf-8").splitlines()
            self.assertEqual(state["status"], "SUCCESS")
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[1])["step"], "MODEL_SEARCH")

    def test_failure_is_recorded(self):
        """Thiếu cấu hình phải ghi FAILED và phát sinh lỗi, không ghi thành công giả."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(FileNotFoundError):
                run_experiment(root)
            states = list((root / "progress_results").glob("*.json"))
            self.assertEqual(len(states), 1)
            state = json.loads(states[0].read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "FAILED")
            self.assertTrue(state["error"])


if __name__ == "__main__":
    unittest.main()
