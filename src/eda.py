import json

import pandas as pd

from .preprocess import GENDER


def summarize(raw: pd.DataFrame, clean: pd.DataFrame, features: list[str]) -> dict:
    numeric = clean[features]
    q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
    iqr = q3 - q1
    flagged = numeric.lt(q1 - 1.5 * iqr) | numeric.gt(q3 + 1.5 * iqr)
    return {
        "rows_raw": len(raw),
        "rows_clean": len(clean),
        "rows_removed": len(raw) - len(clean),
        "missing_raw": raw.isna().sum().to_dict(),
        "gender_counts": clean[GENDER].fillna("Thiếu").value_counts().to_dict(),
        "iqr_flags": flagged.sum().to_dict(),
        "iqr_flagged_rows": json.loads(clean.loc[flagged.any(axis=1)].to_json(orient="records")),
        "summary": json.loads(numeric.describe().round(3).to_json()),
        "pearson": json.loads(numeric.corr().round(3).to_json()),
    }
