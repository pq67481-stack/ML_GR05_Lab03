import pandas as pd
from sklearn.model_selection import KFold

from .evaluate import METRICS, evaluate
from .model import FAMILIES, build_model


def search(train: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = train[config["features"]]
    settings = config["training"]
    seed = config["data"]["seed"]
    splits = list(KFold(settings["folds"], shuffle=True, random_state=seed).split(features))
    records = []
    for family in FAMILIES:
        for k in settings["cluster_counts"]:
            for fold, (fit_index, val_index) in enumerate(splits, start=1):
                fit = features.iloc[fit_index]
                val = features.iloc[val_index]
                model = build_model(family, k, settings, seed).fit(fit)
                metrics, _ = evaluate(model["preprocess"].transform(val), model.predict(val))
                estimator = model["cluster"]
                records.append({
                    "family": family,
                    "k": k,
                    "fold": fold,
                    "converged": getattr(estimator, "converged_", True),
                    **metrics,
                })
    folds = pd.DataFrame(records)
    grouped = folds.groupby(["family", "k"], sort=False)
    scores = grouped[list(METRICS)].agg(["mean", "std"])
    scores.columns = ["_".join(column) for column in scores.columns]
    scores["valid_folds"] = grouped["silhouette"].count()
    scores["converged_folds"] = grouped["converged"].sum()
    scores = scores.reset_index().sort_values(
        ["silhouette_mean", "silhouette_std", "k", "family"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)
    scores["eligible"] = scores["valid_folds"].eq(len(splits)) & scores["converged_folds"].eq(len(splits))
    return scores, folds


def fit_selected(train: pd.DataFrame, scores: pd.DataFrame, config: dict) -> dict:
    eligible = scores.loc[scores["eligible"]]
    if eligible.empty:
        raise ValueError("Không có cấu hình hội tụ với Silhouette hợp lệ ở mọi fold.")
    best = eligible.iloc[0]
    family, k = best["family"], int(best["k"])
    features = config["features"]
    model = build_model(family, k, config["training"], config["data"]["seed"])
    model.fit(train[features])
    if not getattr(model["cluster"], "converged_", True):
        raise RuntimeError("Cấu hình được chọn không hội tụ khi fit lại trên toàn bộ train.")
    return {
        "model": model,
        "family": family,
        "k": k,
        "features": features,
        "ranges": {name: [float(train[name].min()), float(train[name].max())] for name in features},
    }
