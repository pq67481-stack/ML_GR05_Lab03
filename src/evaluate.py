import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
)

from .model import build_model
from .preprocess import GENDER

METRICS = ("silhouette", "davies_bouldin", "calinski_harabasz")


def evaluate(features: np.ndarray, labels: np.ndarray) -> tuple[dict, np.ndarray]:
    counts = pd.Series(labels).value_counts()
    result = {"n_clusters_observed": len(counts), "smallest_cluster": int(counts.min())}
    if not 2 <= len(counts) < len(labels):
        return result | dict.fromkeys(METRICS) | {"negative_fraction": None}, np.full(len(labels), np.nan)
    samples = silhouette_samples(features, labels)
    return result | {
        "silhouette": float(samples.mean()),
        "davies_bouldin": float(davies_bouldin_score(features, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(features, labels)),
        "negative_fraction": float((samples < 0).mean()),
    }, samples


def profile(rows: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    grouped = rows.groupby("cluster", sort=True)
    means = grouped[features].mean().add_suffix("_mean")
    return pd.concat([
        grouped.size().rename("count"),
        (100 * grouped.size() / len(rows)).rename("share_pct"),
        means,
        grouped["silhouette"].mean().rename("silhouette_mean"),
        grouped["silhouette"].apply(lambda s: s.lt(0).sum()).rename("negative_count"),
        pd.crosstab(rows["cluster"], rows[GENDER].fillna("Thiếu")).add_prefix("gender_"),
    ], axis=1).reset_index()


def check_stability(bundle: dict, train: pd.DataFrame, test: pd.DataFrame, config: dict) -> pd.DataFrame:
    features = bundle["features"]
    reference = bundle["model"].predict(test[features])
    settings = config["stability"]
    seed = config["data"]["seed"]
    rng = np.random.default_rng(seed)
    records = []
    for repeat in range(settings["repeats"]):
        sample_seed = int(rng.integers(0, 2**31))
        sample = train.sample(frac=settings["fraction"], random_state=sample_seed)
        model = build_model(bundle["family"], bundle["k"], config["training"], sample_seed)
        model.fit(sample[features])
        labels = model.predict(test[features])
        records.append({
            "repeat": repeat + 1,
            "seed": sample_seed,
            "train_rows": len(sample),
            "test_clusters": len(np.unique(labels)),
            "ari": adjusted_rand_score(reference, labels),
        })
    return pd.DataFrame(records)
