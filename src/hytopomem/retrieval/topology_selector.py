from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from hytopomem.retrieval.topology_features import FEATURE_NAMES


@dataclass
class TopologySelectorArtifact:
    model: object
    feature_names: list[str]
    metadata: dict

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: str | Path) -> "TopologySelectorArtifact":
        with Path(path).open("rb") as handle:
            artifact = pickle.load(handle)
        if not isinstance(artifact, cls):
            raise TypeError(f"unexpected topology selector artifact type: {type(artifact)!r}")
        return artifact


def train_lightgbm_ranker(
    rows: Sequence[Sequence[float]],
    labels: Sequence[int],
    groups: Sequence[int],
    *,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    min_child_samples: int = 10,
    n_jobs: int = 8,
    random_state: int = 13,
):
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError(
            "LightGBM is required for the topology selector. Install it in the active env with `pip install lightgbm`."
        ) from exc

    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        subsample=0.9,
        colsample_bytree=0.9,
        force_col_wise=True,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=-1,
    )
    model.fit(
        rows,
        labels,
        group=list(groups),
    )
    return model


def feature_importance(model, top_k: int = 20) -> list[dict]:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return []
    ranked = sorted(
        zip(FEATURE_NAMES, [float(value) for value in importances]),
        key=lambda item: item[1],
        reverse=True,
    )
    return [{"feature": name, "importance": value} for name, value in ranked[:top_k]]
