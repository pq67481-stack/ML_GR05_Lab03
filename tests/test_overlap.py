"""Kiểm tra chuyển phần chồng lấn về train và bảo vệ dữ liệu mâu thuẫn."""

import unittest
import pandas as pd
from src.preprocessing import check_overlap, resolve_overlap


class TestResolveOverlap(unittest.TestCase):
    """Kiểm tra việc chuyển theo mã khách và tổ hợp đặc trưng."""

    def test_move_overlapping_rows(self):
        """Mã trùng và đặc trưng trùng giữa validation/test đều chuyển về train."""
        columns = ["CustomerID", "Gender", "Age", "Annual Income", "Spending Score"]
        def frame(rows):
            """Tạo bảng nhỏ theo cấu trúc dữ liệu khách hàng."""
            return pd.DataFrame(rows, columns=columns)
        parts = {"train": frame([["a", "Male", 20, 100, 10]]),
                 "validation": frame([["a", "Male", 20, 100, 10], ["b", "Male", 30, 200, 20]]),
                 "test": frame([["c", "Female", 30, 200, 20], ["d", "Male", 40, 300, 30]])}
        resolved, moved, removed = resolve_overlap(parts)
        self.assertEqual(len(moved), 3)
        self.assertEqual(removed, 1)
        self.assertEqual(set(resolved["train"].CustomerID), {"a", "b", "c"})
        self.assertEqual(set(resolved["test"].CustomerID), {"d"})
        summary, _ = check_overlap(resolved)
        self.assertEqual(int(summary.shared_customer_ids.sum()), 0)
        self.assertEqual(int(summary.shared_feature_combinations.sum()), 0)

    def test_conflicting_id_is_rejected(self):
        """Cùng ID nhưng các bản ghi khác nhau phải báo lỗi, không tự xóa."""
        cols = ["CustomerID", "Gender", "Age", "Annual Income", "Spending Score"]
        parts = {"train": pd.DataFrame([["a", "Male", 20, 100, 10]], columns=cols),
                 "validation": pd.DataFrame([["a", "Male", 21, 100, 10]], columns=cols),
                 "test": pd.DataFrame([["b", "Female", 30, 200, 20]], columns=cols)}
        with self.assertRaises(ValueError):
            resolve_overlap(parts)


if __name__ == "__main__":
    unittest.main()
