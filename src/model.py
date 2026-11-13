from pathlib import Path

import joblib
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline

from .preprocess import make_preprocessor

FAMILIES = ("kmeans", "gmm")


def build_model(family: str, k: int, settings: dict, seed: int) -> Pipeline:
    if family == "kmeans":
        estimator = KMeans(n_clusters=k, random_state=seed, **settings[family])
    elif family == "gmm":
        estimator = GaussianMixture(n_components=k, random_state=seed, **settings[family])
    else:
        raise ValueError(f"Thuật toán chưa được hỗ trợ: {family}")
    return Pipeline([("preprocess", make_preprocessor()), ("cluster", estimator)])


def save_model(bundle: dict, path: Path) -> None:
    joblib.dump(bundle, path)


def load_model(path: Path) -> dict:
    return joblib.load(path)
