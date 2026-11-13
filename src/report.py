from pathlib import Path

import pandas as pd


def _table(frame: pd.DataFrame) -> str:
    def format_value(value):
        if pd.isna(value):
            return "—"
        return f"{value:.4f}" if isinstance(value, float) else str(value)

    rows = [
        list(frame.columns),
        ["---"] * len(frame.columns),
        *[[format_value(value) for value in row] for row in frame.itertuples(index=False, name=None)],
    ]
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def write_report(path: Path, run_id: str, eda: dict, scores: pd.DataFrame,
                 metrics: dict, train_profile: pd.DataFrame, test_profile: pd.DataFrame) -> None:
    columns = [
        "family", "k", "silhouette_mean", "silhouette_std", "davies_bouldin_mean",
        "calinski_harabasz_mean", "valid_folds", "converged_folds", "eligible",
    ]
    evaluation = pd.DataFrame.from_dict(
        {part: metrics[part] for part in ("train", "test")}, orient="index"
    ).rename_axis("Tập").reset_index()
    stability = pd.DataFrame([metrics["stability"]])
    sections = [
        f"# Kết quả Lab 03\n\nLần chạy `{run_id}`.",
        f"## Dữ liệu\n\nCó {eda['rows_raw']} dòng gốc và {eda['rows_clean']} dòng sau làm sạch. "
        f"Số dòng bị loại do trùng là {eda['rows_removed']}.",
        "## Tìm cấu hình trên tập train\n\n" + _table(scores[columns]),
        f"## Mô hình lưu\n\n`{metrics['family']}` với `k={metrics['k']}`."
        f" Fit lại trên {metrics['train_rows']} khách và đánh giá trên {metrics['test_rows']} khách test.",
        "## Chỉ số theo tập\n\n" + _table(evaluation),
        "## Độ ổn định khi lấy mẫu lại train\n\n" + _table(stability),
        "## Hồ sơ cụm trên train\n\n" + _table(train_profile),
        "## Hồ sơ cụm trên test\n\n" + _table(test_profile),
        "Công thức, cách đọc và diễn giải lần chạy mặc định có trong "
        "[tài liệu Lab 03](../../docs/Lab_03_descript.md).",
    ]
    path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
