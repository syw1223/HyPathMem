import numpy as np

from hytopomem.models.graph_v2_hyperbolic import expmap0
from hytopomem.retrieval.hyperbolic_topdown_retriever import top_by_lorentz_score
from hytopomem.retrieval.topdown_semantic_retriever import top_by_similarity


def test_top_by_similarity_respects_candidate_subset() -> None:
    node_ids = ["event:a", "event:c"]
    index = {"event:a": 0, "event:b": 1, "event:c": 2}
    matrix = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)

    ranked = top_by_similarity(node_ids, index, matrix, np.asarray([1.0, 0.0], dtype=np.float32), 2)

    assert [node_id for node_id, _score in ranked] == ["event:a", "event:c"]


def test_top_by_lorentz_score_respects_nearest_distance() -> None:
    tangent = np.asarray([[0.1, 0.0], [1.0, 0.0], [0.0, 0.1]], dtype=np.float32)
    import torch

    points = expmap0(torch.tensor(tangent)).numpy()
    query = expmap0(torch.tensor([[0.0, 0.0]], dtype=torch.float32)).numpy()[0]
    ranked = top_by_lorentz_score(
        ["event:a", "event:b", "event:c"],
        {"event:a": 0, "event:b": 1, "event:c": 2},
        points,
        query,
        2,
    )

    assert [node_id for node_id, _score in ranked] == ["event:a", "event:c"]
