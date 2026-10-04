from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from hytopomem.models.graph_v2_hyperbolic import lorentz_distance, lorentz_radius


@dataclass
class StructureBatch:
    child_idx: torch.Tensor
    parent_idx: torch.Tensor
    negative_parent_idx: torch.Tensor


@dataclass
class BranchBatch:
    anchor_idx: torch.Tensor
    positive_idx: torch.Tensor
    negative_idx: torch.Tensor


def edge_contrastive_loss(
    child_points: torch.Tensor,
    parent_points: torch.Tensor,
    negative_parent_points: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    pos_logits = -lorentz_distance(child_points, parent_points)[:, None] / tau
    expanded_child = child_points[:, None, :].expand_as(negative_parent_points)
    neg_logits = -lorentz_distance(expanded_child, negative_parent_points) / tau
    logits = torch.cat([pos_logits, neg_logits], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)


def radial_order_loss(child_points: torch.Tensor, parent_points: torch.Tensor, margin: float) -> torch.Tensor:
    child_radius = lorentz_radius(child_points)
    parent_radius = lorentz_radius(parent_points)
    return torch.relu(margin + parent_radius - child_radius).mean()


def branch_triplet_loss(
    anchor_points: torch.Tensor,
    positive_points: torch.Tensor,
    negative_points: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    pos_dist = lorentz_distance(anchor_points, positive_points)
    neg_dist = lorentz_distance(anchor_points, negative_points)
    return torch.relu(margin + pos_dist - neg_dist).mean()


def sample_structure_batch(
    *,
    edges: list[tuple[int, int, str]],
    parent_pools: dict[str, list[int]],
    batch_size: int,
    num_negatives: int,
    device: torch.device,
) -> StructureBatch:
    sampled = random.choices(edges, k=min(batch_size, len(edges)))
    child_idx = torch.tensor([item[0] for item in sampled], dtype=torch.long, device=device)
    parent_idx = torch.tensor([item[1] for item in sampled], dtype=torch.long, device=device)
    negatives = []
    for _child, parent, parent_type in sampled:
        pool = parent_pools[parent_type]
        row = []
        while len(row) < num_negatives:
            candidate = random.choice(pool)
            if candidate != parent:
                row.append(candidate)
        negatives.append(row)
    negative_parent_idx = torch.tensor(negatives, dtype=torch.long, device=device)
    return StructureBatch(child_idx=child_idx, parent_idx=parent_idx, negative_parent_idx=negative_parent_idx)


def sample_branch_batch(
    *,
    event_to_facts: dict[int, list[int]],
    fact_pool: list[int],
    batch_size: int,
    device: torch.device,
) -> BranchBatch | None:
    eligible_events = [event for event, facts in event_to_facts.items() if len(facts) >= 2]
    if not eligible_events:
        return None
    anchors = []
    positives = []
    negatives = []
    for _ in range(batch_size):
        event = random.choice(eligible_events)
        facts = event_to_facts[event]
        anchor, positive = random.sample(facts, 2)
        negative = random.choice(fact_pool)
        tries = 0
        while negative in facts and tries < 20:
            negative = random.choice(fact_pool)
            tries += 1
        anchors.append(anchor)
        positives.append(positive)
        negatives.append(negative)
    return BranchBatch(
        anchor_idx=torch.tensor(anchors, dtype=torch.long, device=device),
        positive_idx=torch.tensor(positives, dtype=torch.long, device=device),
        negative_idx=torch.tensor(negatives, dtype=torch.long, device=device),
    )
