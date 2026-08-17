from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import math
import pickle
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from common import read_json, resolve_path, write_json
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.retrieval.topology_selector import train_lightgbm_ranker


class HyperCardReasoner(nn.Module):
    def __init__(
        self,
        fact_dim: int,
        card_dim: int,
        role_dim: int,
        hidden_dim: int,
        dropout: float,
        *,
        use_roles: bool = True,
        use_hyperbolic_attention: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_roles = use_roles
        self.use_hyperbolic_attention = use_hyperbolic_attention
        self.hyperbolic_log_scale = nn.Parameter(torch.tensor(-2.0))
        self.fact_encoder = nn.Sequential(
            nn.Linear(fact_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.card_encoder = nn.Sequential(
            nn.Linear(card_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.role_encoder = nn.Linear(role_dim, hidden_dim, bias=False)
        self.fact_to_card_attention = nn.Linear(hidden_dim, 1, bias=False)
        self.card_to_fact_attention = nn.Linear(hidden_dim, 1, bias=False)
        self.card_norm = nn.LayerNorm(hidden_dim)
        self.fact_update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fact_norm = nn.LayerNorm(hidden_dim)
        self.fact_scorer = nn.Linear(hidden_dim, 1)
        self.card_scorer = nn.Linear(hidden_dim, 1)

    def forward(self, example: dict[str, object]) -> dict[str, torch.Tensor]:
        fact_x = example["fact_features"]
        card_x = example["card_features"]
        edge_fact_indices = example["edge_fact_indices"]
        edge_card_indices = example["edge_card_indices"]
        edge_roles = example["edge_roles"]
        fact_lorentz = example.get("fact_lorentz")
        fact_hidden = self.fact_encoder(fact_x)
        if card_x.shape[0] == 0 or edge_fact_indices.numel() == 0:
            zeros = torch.zeros_like(fact_hidden)
            updated = self.fact_norm(fact_hidden)
            fact_scores = self.fact_scorer(updated).squeeze(-1)
            empty = fact_scores.new_zeros((0,))
            propagation = torch.stack(
                [
                    fact_scores,
                    zeros.norm(dim=-1),
                    zeros.norm(dim=-1),
                    fact_scores.new_zeros(fact_scores.shape),
                    fact_scores.new_zeros(fact_scores.shape),
                    fact_scores.new_zeros(fact_scores.shape),
                    fact_scores.new_zeros(fact_scores.shape),
                    F.cosine_similarity(fact_hidden, updated, dim=-1),
                    fact_scores.new_zeros(fact_scores.shape),
                    fact_scores.new_zeros(fact_scores.shape),
                ],
                dim=-1,
            )
            return {
                "fact_scores": fact_scores,
                "card_scores": empty,
                "propagation_features": propagation,
                "updated_fact_hidden": updated,
            }

        card_seed = self.card_encoder(card_x)
        member_hidden = fact_hidden[edge_fact_indices]
        role_hidden = (
            self.role_encoder(edge_roles)
            if self.use_roles
            else torch.zeros_like(member_hidden)
        )
        fact_to_card_logits = self.fact_to_card_attention(
            torch.tanh(member_hidden + card_seed[edge_card_indices] + role_hidden)
        ).squeeze(-1)
        edge_hyp_distance = fact_to_card_logits.new_zeros(fact_to_card_logits.shape)
        if self.use_hyperbolic_attention and fact_lorentz is not None:
            card_lorentz = lorentz_card_centroids(
                fact_lorentz,
                edge_fact_indices,
                edge_card_indices,
                card_x.shape[0],
            )
            edge_hyp_distance = lorentz_distance(
                fact_lorentz[edge_fact_indices],
                card_lorentz[edge_card_indices],
            )
            geometry_scale = F.softplus(self.hyperbolic_log_scale)
            fact_to_card_logits = fact_to_card_logits - geometry_scale * edge_hyp_distance
        fact_to_card_weights = segment_softmax(
            fact_to_card_logits,
            edge_card_indices,
            card_x.shape[0],
        )
        card_aggregate = fact_hidden.new_zeros((card_x.shape[0], self.hidden_dim))
        card_aggregate.index_add_(
            0,
            edge_card_indices,
            fact_to_card_weights[:, None] * (member_hidden + role_hidden),
        )
        card_hidden = self.card_norm(card_seed + card_aggregate)
        card_scores = self.card_scorer(card_hidden).squeeze(-1)

        messages = card_hidden[edge_card_indices] + role_hidden
        card_to_fact_logits = self.card_to_fact_attention(
            torch.tanh(fact_hidden[edge_fact_indices] + messages)
        ).squeeze(-1)
        if self.use_hyperbolic_attention and fact_lorentz is not None:
            card_to_fact_logits = (
                card_to_fact_logits
                - F.softplus(self.hyperbolic_log_scale) * edge_hyp_distance
            )
        card_to_fact_weights = segment_softmax(
            card_to_fact_logits,
            edge_fact_indices,
            fact_hidden.shape[0],
        )
        aggregate_messages = torch.zeros_like(fact_hidden)
        aggregate_messages.index_add_(
            0,
            edge_fact_indices,
            card_to_fact_weights[:, None] * messages,
        )
        attention_max = segment_max(
            card_to_fact_weights,
            edge_fact_indices,
            fact_hidden.shape[0],
            fill_value=0.0,
        )
        incident_card_scores = card_scores[edge_card_indices]
        card_score_max = segment_max(
            incident_card_scores,
            edge_fact_indices,
            fact_hidden.shape[0],
            fill_value=0.0,
        )
        card_score_sum = fact_hidden.new_zeros((fact_hidden.shape[0],))
        card_score_sum.index_add_(0, edge_fact_indices, incident_card_scores)
        membership_count = fact_hidden.new_zeros((fact_hidden.shape[0],))
        membership_count.index_add_(
            0,
            edge_fact_indices,
            torch.ones_like(incident_card_scores),
        )
        card_score_mean = card_score_sum / membership_count.clamp_min(1.0)
        hyp_distance_sum = fact_hidden.new_zeros((fact_hidden.shape[0],))
        hyp_distance_sum.index_add_(0, edge_fact_indices, edge_hyp_distance)
        hyp_distance_mean = hyp_distance_sum / membership_count.clamp_min(1.0)
        hyp_distance_min = segment_min(
            edge_hyp_distance,
            edge_fact_indices,
            fact_hidden.shape[0],
            fill_value=0.0,
        )

        delta = self.fact_update(aggregate_messages)
        updated = self.fact_norm(fact_hidden + delta)
        fact_scores = self.fact_scorer(updated).squeeze(-1)
        propagation = torch.stack(
            [
                fact_scores,
                aggregate_messages.norm(dim=-1),
                delta.norm(dim=-1),
                attention_max,
                card_score_max,
                card_score_mean,
                membership_count,
                F.cosine_similarity(fact_hidden, updated, dim=-1),
                hyp_distance_mean,
                hyp_distance_min,
            ],
            dim=-1,
        )
        return {
            "fact_scores": fact_scores,
            "card_scores": card_scores,
            "propagation_features": propagation,
            "updated_fact_hidden": updated,
        }


def segment_softmax(
    values: torch.Tensor,
    indices: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    maxima = segment_max(values, indices, num_segments, fill_value=-torch.inf)
    exponentials = torch.exp(values - maxima[indices])
    denominators = values.new_zeros((num_segments,))
    denominators.index_add_(0, indices, exponentials)
    return exponentials / denominators[indices].clamp_min(1e-12)


def segment_max(
    values: torch.Tensor,
    indices: torch.Tensor,
    num_segments: int,
    *,
    fill_value: float,
) -> torch.Tensor:
    output = values.new_full((num_segments,), fill_value)
    output.scatter_reduce_(0, indices, values, reduce="amax", include_self=True)
    return output


def segment_min(
    values: torch.Tensor,
    indices: torch.Tensor,
    num_segments: int,
    *,
    fill_value: float,
) -> torch.Tensor:
    output = values.new_full((num_segments,), torch.inf)
    output.scatter_reduce_(0, indices, values, reduce="amin", include_self=True)
    return torch.where(torch.isinf(output), output.new_full(output.shape, fill_value), output)


def lorentz_card_centroids(
    fact_points: torch.Tensor,
    edge_fact_indices: torch.Tensor,
    edge_card_indices: torch.Tensor,
    num_cards: int,
) -> torch.Tensor:
    spatial = fact_points[edge_fact_indices, 1:]
    spatial_sum = spatial.new_zeros((num_cards, spatial.shape[-1]))
    spatial_sum.index_add_(0, edge_card_indices, spatial)
    counts = spatial.new_zeros((num_cards,))
    counts.index_add_(0, edge_card_indices, torch.ones_like(edge_card_indices, dtype=spatial.dtype))
    spatial_mean = spatial_sum / counts[:, None].clamp_min(1.0)
    time = torch.sqrt(1.0 + (spatial_mean * spatial_mean).sum(dim=-1, keepdim=True))
    return torch.cat([time, spatial_mean], dim=-1)


def lorentz_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    inner = -left[:, 0] * right[:, 0] + (left[:, 1:] * right[:, 1:]).sum(dim=-1)
    return torch.acosh((-inner).clamp_min(1.0 + 1e-6))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="outputs/v4_0/query_induced_hypergraphs_ctx50.pkl.gz")
    parser.add_argument(
        "--candidates",
        default="outputs/v3_9_query_cards/qwen3_card_annotated_base100_paths_v3_clean.json",
    )
    parser.add_argument("--graph", default="outputs/graphs/locomo_graph_v3_6b_qwen_all.json")
    parser.add_argument("--output-dir", default="outputs/eval/cv/v4_0_hypercard_mp")
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--card-loss-weight", type=float, default=0.30)
    parser.add_argument("--consistency-weight", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-estimators", type=int, default=80)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument("--disable-roles", action="store_true")
    parser.add_argument("--disable-card-loss", action="store_true")
    parser.add_argument("--use-hyperbolic-attention", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    torch.set_num_threads(min(8, max(1, torch.get_num_threads())))
    payload = load_dataset(resolve_path(args.dataset))
    examples = payload["examples"]
    metadata = payload["metadata"]
    candidates = read_json(resolve_path(args.candidates))
    item_map = {item["question_id"]: item for item in candidates}
    graph = JsonGraphStore().load(resolve_path(args.graph))
    cv = load_cv_helpers()
    all_conversations = ordered_conversations(examples)
    conversations = list(all_conversations)
    if args.max_folds:
        conversations = conversations[: args.max_folds]
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    aggregate = {"h2_neural": defaultdict(list), "h2_mp_lgbm": defaultdict(list), "base_lgbm": defaultdict(list)}
    fold_rows = []
    for fold_index, test_conversation in enumerate(conversations):
        fold_started = time.perf_counter()
        test_index = all_conversations.index(test_conversation)
        validation_conversation = all_conversations[(test_index + 1) % len(all_conversations)]
        train_selection = [
            example
            for example in examples
            if example["conversation_id"] not in {test_conversation, validation_conversation}
        ]
        validation = [
            example for example in examples if example["conversation_id"] == validation_conversation
        ]
        train_full = [
            example for example in examples if example["conversation_id"] != test_conversation
        ]
        test = [
            example for example in examples if example["conversation_id"] == test_conversation
        ]

        selection_scalers = fit_scalers(train_selection)
        selection_model = create_model(metadata, args, device)
        best_epoch, history = select_epoch(
            selection_model,
            train_selection,
            validation,
            selection_scalers,
            args,
            device,
            fold_seed=args.seed + fold_index * 17,
        )
        full_scalers = fit_scalers(train_full)
        model = create_model(metadata, args, device)
        train_fixed_epochs(
            model,
            train_full,
            full_scalers,
            args,
            device,
            epochs=best_epoch,
            fold_seed=args.seed + fold_index * 17 + 7,
        )

        neural_scores = score_examples(model, test, full_scalers, device)
        neural_items = rerank_items(item_map, test, neural_scores, "h2_neural")
        train_mp = propagation_rows(model, train_full, full_scalers, device)
        test_mp = propagation_rows(model, test, full_scalers, device)
        lgbm_model = train_lightgbm_ranker(
            train_mp["rows"],
            train_mp["labels"],
            train_mp["groups"],
            n_estimators=args.n_estimators,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=10,
            n_jobs=8,
            random_state=args.seed + fold_index,
        )
        lgbm_scores = predict_lgbm(lgbm_model, test_mp)
        lgbm_items = rerank_items(item_map, test, lgbm_scores, "h2_mp_lgbm")

        base_train = base_rows(train_full)
        base_test = base_rows(test)
        base_model = train_lightgbm_ranker(
            base_train["rows"],
            base_train["labels"],
            base_train["groups"],
            n_estimators=args.n_estimators,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=10,
            n_jobs=8,
            random_state=args.seed + fold_index,
        )
        base_scores = predict_lgbm(base_model, base_test)
        base_items = rerank_items(item_map, test, base_scores, "base_lgbm")

        fold = {
            "fold": fold_index,
            "test_conversation": test_conversation,
            "validation_conversation": validation_conversation,
            "best_epoch": best_epoch,
            "history": history,
            "methods": {},
        }
        for method, method_items in [
            ("base_lgbm", base_items),
            ("h2_neural", neural_items),
            ("h2_mp_lgbm", lgbm_items),
        ]:
            fold["methods"][method] = {}
            for k in (5, 20):
                evaluation = cv.evaluate_items(graph, method_items, k, method)
                fold["methods"][method][f"top{k}"] = evaluation["summary"]
                aggregate[method][k].extend(evaluation["per_question"])
                fold_dir = output_dir / f"fold_{fold_index:02d}_{test_conversation}"
                write_json(evaluation, fold_dir / f"{method}_top{k}_eval.json")
            write_json(method_items, output_dir / f"fold_{fold_index:02d}_{test_conversation}" / f"{method}_paths.json")
        write_json(fold, output_dir / f"fold_{fold_index:02d}_{test_conversation}" / "fold_summary.json")
        fold_rows.append(fold)
        print(
            f"fold={fold_index} test={test_conversation} val={validation_conversation} "
            f"best_epoch={best_epoch} elapsed={time.perf_counter() - fold_started:.1f}s",
            flush=True,
        )

    summary = {
        "method": "V4.0 Query-induced HyperCard Message Passing",
        "dataset": str(resolve_path(args.dataset)),
        "candidates": str(resolve_path(args.candidates)),
        "device": str(device),
        "config": vars(args),
        "aggregate": {
            method: {
                f"top{k}": cv.summarize_rows(rows)
                for k, rows in by_k.items()
            }
            for method, by_k in aggregate.items()
        },
        "frozen_mainline": {
            "top5": {"hit": 0.8092, "recall": 0.7473, "full_cover": 0.6934},
        },
        "folds": fold_rows,
    }
    write_json(summary, output_dir / "hypercard_mp_summary.json")
    (output_dir / "hypercard_mp_summary.md").write_text(render_summary(summary), encoding="utf-8")
    print(render_summary(summary))


def create_model(metadata: dict, args, device: torch.device) -> HyperCardReasoner:
    model = HyperCardReasoner(
        fact_dim=len(metadata["base_feature_names"]),
        card_dim=len(metadata["card_feature_names"]),
        role_dim=len(metadata["role_names"]),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        use_roles=not args.disable_roles,
        use_hyperbolic_attention=args.use_hyperbolic_attention,
    )
    return model.to(device)


def select_epoch(
    model: HyperCardReasoner,
    train_examples: list[dict],
    validation_examples: list[dict],
    scalers: dict,
    args,
    device: torch.device,
    fold_seed: int,
) -> tuple[int, list[dict]]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    best_epoch = 1
    best_utility = -1.0
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_examples, scalers, optimizer, args, device, fold_seed + epoch)
        metrics = validation_metrics(
            model,
            validation_examples,
            scalers,
            device,
            batch_size=args.batch_size,
            k=5,
        )
        history.append({"epoch": epoch, "loss": loss, **metrics})
        if metrics["utility_at_5"] > best_utility + 1e-5:
            best_utility = metrics["utility_at_5"]
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    return best_epoch, history


def train_fixed_epochs(
    model: HyperCardReasoner,
    examples: list[dict],
    scalers: dict,
    args,
    device: torch.device,
    *,
    epochs: int,
    fold_seed: int,
) -> None:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    for epoch in range(1, epochs + 1):
        train_one_epoch(model, examples, scalers, optimizer, args, device, fold_seed + epoch)


def train_one_epoch(model, examples, scalers, optimizer, args, device, seed: int) -> float:
    model.train()
    order = list(range(len(examples)))
    random.Random(seed).shuffle(order)
    total_loss = 0.0
    used = 0
    for start in range(0, len(order), args.batch_size):
        raw_batch = [examples[index] for index in order[start : start + args.batch_size]]
        batch = tensorize_batch(raw_batch, scalers, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        loss = batch_query_loss(output, batch, args)
        if loss is None:
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        used += len(raw_batch)
    return total_loss / max(used, 1)


def batch_query_loss(output: dict, batch: dict, args) -> torch.Tensor | None:
    query_losses = []
    for fact_slice, card_slice in zip(batch["fact_slices"], batch["card_slices"]):
        fact_loss = pairwise_ranking_loss(
            output["fact_scores"][fact_slice],
            batch["fact_labels"][fact_slice],
        )
        if fact_loss is None:
            continue
        total = fact_loss
        card_scores = output["card_scores"][card_slice]
        card_labels = batch["card_labels"][card_slice]
        if (
            not args.disable_card_loss
            and card_scores.numel()
            and card_labels.sum() > 0
            and (1.0 - card_labels).sum() > 0
        ):
            card_loss = pairwise_ranking_loss(card_scores, card_labels)
            if card_loss is not None:
                total = total + args.card_loss_weight * card_loss
        query_losses.append(total)
    if not query_losses:
        return None
    total_loss = torch.stack(query_losses).mean()
    if output["card_scores"].numel() and batch["edge_card_indices"].numel():
        member_fact_scores = torch.sigmoid(output["fact_scores"][batch["edge_fact_indices"]])
        member_max = segment_max(
            member_fact_scores,
            batch["edge_card_indices"],
            output["card_scores"].shape[0],
            fill_value=0.0,
        )
        total_loss = total_loss + args.consistency_weight * F.mse_loss(
            torch.sigmoid(output["card_scores"]),
            member_max,
        )
    return total_loss


def pairwise_ranking_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor | None:
    positive = scores[labels > 0.5]
    negative = scores[labels <= 0.5]
    if positive.numel() == 0 or negative.numel() == 0:
        return None
    if negative.numel() > 30:
        negative = torch.topk(negative, k=30).values
    return F.softplus(negative[None, :] - positive[:, None]).mean()


def validation_metrics(model, examples, scalers, device, batch_size: int, k: int) -> dict:
    model.eval()
    hits = []
    recalls = []
    full_covers = []
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            raw_batch = examples[start : start + batch_size]
            batch = tensorize_batch(raw_batch, scalers, device)
            output = model(batch)
            for raw, fact_slice in zip(raw_batch, batch["fact_slices"]):
                labels = np.asarray(raw["fact_labels"], dtype=np.float32)
                positives = int(labels.sum())
                if positives <= 0:
                    continue
                scores = output["fact_scores"][fact_slice]
                top_indices = torch.topk(scores, k=min(k, len(labels))).indices.cpu().numpy()
                matched = float(labels[top_indices].sum())
                hits.append(float(matched > 0))
                recalls.append(matched / positives)
                full_covers.append(float(matched >= positives))
    hit = sum(hits) / max(len(hits), 1)
    recall = sum(recalls) / max(len(recalls), 1)
    full_cover = sum(full_covers) / max(len(full_covers), 1)
    return {
        "validation_hit_at_5": hit,
        "validation_recall_at_5": recall,
        "validation_full_cover_at_5": full_cover,
        "utility_at_5": 0.20 * hit + 0.35 * recall + 0.45 * full_cover,
    }


def score_examples(model, examples, scalers, device) -> dict[str, list[float]]:
    model.eval()
    scores = {}
    with torch.no_grad():
        for start in range(0, len(examples), 64):
            raw_batch = examples[start : start + 64]
            batch = tensorize_batch(raw_batch, scalers, device)
            output = model(batch)
            for raw, fact_slice in zip(raw_batch, batch["fact_slices"]):
                scores[raw["question_id"]] = output["fact_scores"][fact_slice].cpu().tolist()
    return scores


def propagation_rows(model, examples, scalers, device) -> dict:
    model.eval()
    rows = []
    labels = []
    groups = []
    qids = []
    with torch.no_grad():
        for start in range(0, len(examples), 64):
            raw_batch = examples[start : start + 64]
            batch = tensorize_batch(raw_batch, scalers, device)
            output = model(batch)
            for raw, fact_slice in zip(raw_batch, batch["fact_slices"]):
                propagation = output["propagation_features"][fact_slice].cpu().numpy()
                base = np.asarray(raw["fact_features"], dtype=np.float64)
                rows.extend(np.concatenate([base, propagation], axis=1).tolist())
                labels.extend(np.asarray(raw["fact_labels"], dtype=np.int32).tolist())
                groups.append(len(base))
                qids.append(raw["question_id"])
    return {"rows": rows, "labels": labels, "groups": groups, "qids": qids}


def base_rows(examples: list[dict]) -> dict:
    rows = []
    labels = []
    groups = []
    qids = []
    for raw in examples:
        features = np.asarray(raw["fact_features"], dtype=np.float64)
        rows.extend(features.tolist())
        labels.extend(np.asarray(raw["fact_labels"], dtype=np.int32).tolist())
        groups.append(len(features))
        qids.append(raw["question_id"])
    return {"rows": rows, "labels": labels, "groups": groups, "qids": qids}


def predict_lgbm(model, payload: dict) -> dict[str, list[float]]:
    rows = np.asarray(payload["rows"], dtype=np.float64)
    predictions = model.predict(rows)
    output = {}
    offset = 0
    for qid, group in zip(payload["qids"], payload["groups"]):
        output[qid] = [float(value) for value in predictions[offset : offset + group]]
        offset += group
    return output


def rerank_items(item_map, examples, scores_by_qid, method: str) -> list[dict]:
    output = []
    for example in examples:
        item = item_map[example["question_id"]]
        scores = scores_by_qid[example["question_id"]]
        ranked = sorted(zip(item.get("paths", []), scores), key=lambda row: row[1], reverse=True)
        copied = dict(item)
        paths = []
        for path, score in ranked:
            copied_path = dict(path)
            path_scores = dict(copied_path.get("scores", {}))
            path_scores[method] = float(score)
            copied_path["scores"] = path_scores
            paths.append(copied_path)
        copied["paths"] = paths
        metadata = dict(copied.get("metadata", {}))
        metadata["method"] = method
        copied["metadata"] = metadata
        output.append(copied)
    return output


def tensorize_batch(raw_batch: list[dict], scalers: dict, device: torch.device) -> dict:
    fact_feature_rows = []
    fact_labels = []
    card_feature_rows = []
    card_labels = []
    edge_fact_indices = []
    edge_card_indices = []
    edge_roles = []
    fact_slices = []
    card_slices = []
    fact_offset = 0
    card_offset = 0
    role_dim = 0
    fact_lorentz_rows = []
    for raw in raw_batch:
        facts = normalize(
            np.asarray(raw["fact_features"], dtype=np.float32),
            scalers["fact"],
        )
        cards = raw["cards"]
        fact_feature_rows.append(facts)
        if raw.get("fact_lorentz") is not None:
            fact_lorentz_rows.append(np.asarray(raw["fact_lorentz"], dtype=np.float32))
        fact_labels.extend(raw["fact_labels"])
        fact_slices.append(slice(fact_offset, fact_offset + len(facts)))
        if cards:
            card_features = normalize(
                np.stack([np.asarray(card["features"], dtype=np.float32) for card in cards]),
                scalers["card"],
            )
            card_feature_rows.append(card_features)
            card_labels.extend(card["label"] for card in cards)
        card_slices.append(slice(card_offset, card_offset + len(cards)))
        for local_card_index, card in enumerate(cards):
            for member in card["members"]:
                roles = np.asarray(member["roles"], dtype=np.float32)
                role_dim = max(role_dim, int(roles.shape[0]))
                edge_fact_indices.append(fact_offset + member["fact_index"])
                edge_card_indices.append(card_offset + local_card_index)
                edge_roles.append(roles)
        fact_offset += len(facts)
        card_offset += len(cards)
    if card_feature_rows:
        card_matrix = np.concatenate(card_feature_rows, axis=0)
    else:
        card_matrix = np.zeros((0, scalers["card"]["mean"].shape[0]), dtype=np.float32)
    if edge_roles:
        role_matrix = np.stack(edge_roles)
    else:
        role_matrix = np.zeros((0, role_dim or 1), dtype=np.float32)
    output = {
        "fact_features": torch.tensor(
            np.concatenate(fact_feature_rows, axis=0),
            dtype=torch.float32,
            device=device,
        ),
        "fact_labels": torch.tensor(fact_labels, dtype=torch.float32, device=device),
        "card_features": torch.tensor(card_matrix, dtype=torch.float32, device=device),
        "card_labels": torch.tensor(card_labels, dtype=torch.float32, device=device),
        "edge_fact_indices": torch.tensor(edge_fact_indices, dtype=torch.long, device=device),
        "edge_card_indices": torch.tensor(edge_card_indices, dtype=torch.long, device=device),
        "edge_roles": torch.tensor(role_matrix, dtype=torch.float32, device=device),
        "fact_slices": fact_slices,
        "card_slices": card_slices,
    }
    if fact_lorentz_rows:
        output["fact_lorentz"] = torch.tensor(
            np.concatenate(fact_lorentz_rows, axis=0),
            dtype=torch.float32,
            device=device,
        )
    else:
        output["fact_lorentz"] = None
    return output


def fit_scalers(examples: list[dict]) -> dict:
    fact_rows = np.concatenate([np.asarray(example["fact_features"], dtype=np.float32) for example in examples], axis=0)
    card_rows = [
        np.asarray(card["features"], dtype=np.float32)
        for example in examples
        for card in example["cards"]
    ]
    if card_rows:
        card_matrix = np.stack(card_rows)
    else:
        card_matrix = np.zeros((1, 4), dtype=np.float32)
    return {"fact": scaler(fact_rows), "card": scaler(card_matrix)}


def scaler(matrix: np.ndarray) -> dict:
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std[std < 1e-6] = 1.0
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def normalize(matrix: np.ndarray, stats: dict) -> np.ndarray:
    return (matrix - stats["mean"]) / stats["std"]


def load_dataset(path: Path) -> dict:
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def ordered_conversations(examples: list[dict]) -> list[str]:
    output = []
    seen = set()
    for example in examples:
        conversation = example["conversation_id"]
        if conversation not in seen:
            seen.add(conversation)
            output.append(conversation)
    return output


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def render_summary(summary: dict) -> str:
    lines = [
        "# V4.0 Query-induced HyperCard Message Passing",
        "",
        f"- Device: {summary['device']}",
        "",
        "| Method | K | Hit | Recall | FullCover |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, by_k in summary["aggregate"].items():
        for key, row in by_k.items():
            lines.append(
                f"| {method} | {key} | {row['hit']:.4f} | "
                f"{row['recall']:.4f} | {row['full_cover']:.4f} |"
            )
    lines.extend(
        [
            "",
            "Frozen V3.9 Membership + Quality Top5: "
            "`0.8092 / 0.7473 / 0.6934`.",
            "",
        ]
    )
    return "\n".join(lines)


def load_cv_helpers():
    path = Path(__file__).resolve().parent / "30_run_loco_cv_selector.py"
    spec = importlib.util.spec_from_file_location("loco_cv_v4_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    main()
