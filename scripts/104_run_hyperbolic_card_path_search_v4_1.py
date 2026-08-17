from __future__ import annotations

import argparse
import gzip
import json
import math
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from common import read_json, resolve_path, write_json
from hytopomem.eval.retrieval_metrics import evaluate_item, normalize_evidence_id, summarize
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.retrieval.hyperbolic_topdown_retriever import load_hyperbolic_router


ROLE_ALIASES = {
    "old_state": {"old_state"},
    "new_state": {"new_state", "state_value", "decision", "progress"},
    "reason": {"reason_or_trigger", "reason", "evidence"},
    "preference": {"preference_value", "polarity"},
    "constraint": {"constraint", "exception", "plan_goal"},
    "time": {"temporal_scope"},
    "context": {"context", "location", "evidence"},
}

DEFAULT_DEMAND = {"context"}


@dataclass
class CardNode:
    card_id: str
    rank: int
    relation_type: str
    entity: str
    aspect: str
    summary: str
    confidence: float
    card_ce: float
    fact_ids: list[str]
    roles_by_fact: dict[str, set[str]]
    roles: set[str]
    event_ids: set[str] = field(default_factory=set)
    topic_ids: set[str] = field(default_factory=set)
    point: np.ndarray | None = None
    radius_penalty: float = 0.0


@dataclass
class PathState:
    card_ids: tuple[str, ...]
    score: float
    covered_roles: frozenset[str]
    facts: tuple[str, ...]
    components: dict[str, float]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default="outputs/graphs/locomo_graph_v3_6b_qwen_all.json")
    parser.add_argument(
        "--cards",
        default="outputs/v3_9_query_cards/qwen3_card_annotated_base100_paths_v3_clean.json",
    )
    parser.add_argument(
        "--cardce-paths",
        default="outputs/eval/v3_9_cardce_guided_ctx50/cardce_guided_topk_paths.json",
    )
    parser.add_argument(
        "--hypergraph-cache",
        default="outputs/v4_0/query_induced_hypergraphs_ctx50_mainline_lorentz.pkl.gz",
    )
    parser.add_argument("--trained-card-checkpoint", default="")
    parser.add_argument("--trained-card-embedding-cache", default="outputs/embeddings/graph_v4_1_card_hybrid_fact_card_event_topic.npz")
    parser.add_argument("--trained-card-device", default="cpu")
    parser.add_argument("--baseline-cv-dir", default="outputs/eval/cv/nary_v3_6c_selector_base100_top20")
    parser.add_argument("--baseline-method", default="base_completion_nary_point_set_features")
    parser.add_argument("--output-dir", default="outputs/eval/v4_1_hyperbolic_card_path_search")
    parser.add_argument("--topk", type=int, nargs="+", default=[5, 20])
    parser.add_argument("--max-cards", type=int, default=3)
    parser.add_argument("--beam-size", type=int, default=8)
    parser.add_argument("--start-cards", type=int, default=8)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--max-facts-per-card", type=int, default=4)
    parser.add_argument("--role-beta", type=float, default=1.20)
    parser.add_argument("--chain-eta", type=float, default=0.35)
    parser.add_argument("--redundancy-delta", type=float, default=0.08)
    parser.add_argument("--cost-lambda", type=float, default=0.04)
    parser.add_argument("--hyp-gamma", type=float, default=0.08)
    parser.add_argument("--lgbm-mu", type=float, default=0.18)
    args = parser.parse_args()

    graph = JsonGraphStore().load(resolve_path(args.graph))
    items = read_json(resolve_path(args.cards))
    cardce_scores = load_cardce_scores(resolve_path(args.cardce_paths))
    lorentz = load_lorentz_by_question(resolve_path(args.hypergraph_cache))
    trained_points = (
        load_trained_point_map(
            resolve_path(args.trained_card_checkpoint),
            resolve_path(args.trained_card_embedding_cache),
            args.trained_card_device,
        )
        if args.trained_card_checkpoint
        else {}
    )
    baseline = load_cv_paths(resolve_path(args.baseline_cv_dir), args.baseline_method)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = {
        "v4_1_a_cardpath_nohyp": [],
        "v4_1_b_cardpath_hyp": [],
        "v4_1_c_cardpath_hyp_lgbm_prior": [],
    }
    path_debug = {name: [] for name in methods}

    for index, item in enumerate(items, start=1):
        qid = str(item["question_id"])
        point_map = dict(lorentz.get(qid, {}))
        point_map.update({node_id: point for node_id, point in trained_points.items() if node_id.startswith(qid.split(":q", 1)[0])})
        cards = build_cards(item, cardce_scores, point_map)
        demand = infer_role_demand(item.get("question", ""), cards)
        variants = {
            "v4_1_a_cardpath_nohyp": dict(use_hyp=False, lgbm_mu=0.0),
            "v4_1_b_cardpath_hyp": dict(use_hyp=True, lgbm_mu=0.0),
            "v4_1_c_cardpath_hyp_lgbm_prior": dict(use_hyp=True, lgbm_mu=args.lgbm_mu),
        }
        for method, variant in variants.items():
            path_states = beam_search_cards(item, cards, demand, args, **variant)
            ranked_paths = paths_from_states(item, cards, path_states, max(args.topk), args, lgbm_mu=variant["lgbm_mu"])
            methods[method].append(make_item(item, ranked_paths, method, path_states, demand))
            path_debug[method].append(path_summary(item, graph, cards, path_states, demand))
        if index % 250 == 0 or index == len(items):
            print(f"v4.1 card paths {index}/{len(items)}", flush=True)

    payload = {
        "metadata": {
            "graph": str(resolve_path(args.graph)),
            "cards": str(resolve_path(args.cards)),
            "cardce_paths": str(resolve_path(args.cardce_paths)),
            "hypergraph_cache": str(resolve_path(args.hypergraph_cache)),
            "trained_card_checkpoint": str(resolve_path(args.trained_card_checkpoint)) if args.trained_card_checkpoint else "",
            "trained_card_embedding_cache": str(resolve_path(args.trained_card_embedding_cache)) if args.trained_card_checkpoint else "",
            "baseline_cv_dir": str(resolve_path(args.baseline_cv_dir)),
            "baseline_method": args.baseline_method,
            "max_cards": args.max_cards,
            "beam_size": args.beam_size,
            "start_cards": args.start_cards,
            "neighbors": args.neighbors,
            "max_facts_per_card": args.max_facts_per_card,
            "role_beta": args.role_beta,
            "chain_eta": args.chain_eta,
            "redundancy_delta": args.redundancy_delta,
            "cost_lambda": args.cost_lambda,
            "hyp_gamma": args.hyp_gamma,
            "lgbm_mu": args.lgbm_mu,
        },
        "retrieval": {},
        "path_analysis": {},
        "fixes_vs_baseline": {},
    }

    all_methods = dict(methods)
    if baseline:
        all_methods["h0_membership_quality_lgbm"] = baseline

    for method, method_items in all_methods.items():
        write_json(method_items, output_dir / f"{method}_paths.json")
        payload["retrieval"][method] = {}
        for k in args.topk:
            rows = [evaluate_item(graph, item, k) for item in method_items]
            payload["retrieval"][method][f"top{k}"] = summarize(rows)
            write_json(
                {"method": method, "k": k, "summary": payload["retrieval"][method][f"top{k}"], "per_question": [row.__dict__ for row in rows]},
                output_dir / f"{method}_top{k}_eval.json",
            )

    for method, summaries in path_debug.items():
        payload["path_analysis"][method] = aggregate_path_summaries(summaries)
        write_json(summaries, output_dir / f"{method}_path_analysis.json")

    if baseline:
        for method, method_items in methods.items():
            payload["fixes_vs_baseline"][method] = {}
            for k in args.topk:
                payload["fixes_vs_baseline"][method][f"top{k}"] = compare_items(graph, baseline, method_items, k)

    write_json(payload, output_dir / "v4_1_card_path_search_summary.json")
    (output_dir / "V4_1_CARD_PATH_SEARCH_SUMMARY.md").write_text(render_markdown(payload), encoding="utf-8")
    print(render_markdown(payload))
    print(f"wrote {output_dir / 'V4_1_CARD_PATH_SEARCH_SUMMARY.md'}")


def load_cardce_scores(path: Path) -> dict[tuple[str, str], float]:
    scores: dict[tuple[str, str], float] = {}
    for item in read_json(path):
        qid = str(item["question_id"])
        for evidence_path in item.get("paths", []):
            metadata = evidence_path.get("metadata", {})
            card_id = str(metadata.get("nary_hyperedge_id") or "")
            if not card_id:
                continue
            score = float(evidence_path.get("scores", {}).get("v3_9_card_ce", metadata.get("v3_9_cardce_score", 0.0)) or 0.0)
            scores[(qid, card_id)] = max(scores.get((qid, card_id), float("-inf")), score)
    return scores


def load_lorentz_by_question(path: Path) -> dict[str, dict[str, np.ndarray]]:
    if not path.exists():
        return {}
    with gzip.open(path, "rb") as handle:
        payload = pickle.load(handle)
    out = {}
    for example in payload.get("examples", []):
        points = example.get("fact_lorentz")
        if points is None:
            continue
        out[str(example["question_id"])] = {
            fact_id: np.asarray(points[index], dtype=np.float32)
            for index, fact_id in enumerate(example.get("fact_ids", []))
        }
    return out


def load_trained_point_map(checkpoint_path: Path, embedding_cache_path: Path, device_name: str) -> dict[str, np.ndarray]:
    payload = np.load(embedding_cache_path, allow_pickle=False)
    node_ids = [str(item) for item in payload["node_ids"]]
    embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
    device = torch.device(device_name if (not device_name.startswith("cuda") or torch.cuda.is_available()) else "cpu")
    router = load_hyperbolic_router(checkpoint_path, device)
    router.model.eval()
    rows = []
    with torch.no_grad():
        for start in range(0, len(embeddings), 2048):
            batch = torch.tensor(embeddings[start : start + 2048], dtype=torch.float32, device=device)
            rows.append(router.model(batch).detach().cpu().numpy())
    points = np.concatenate(rows, axis=0).astype(np.float32)
    return dict(zip(node_ids, points))


def build_cards(item: dict, cardce_scores: dict[tuple[str, str], float], point_map: dict[str, np.ndarray]) -> dict[str, CardNode]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for path in item.get("paths", []):
        metadata = path.get("metadata", {})
        card_id = str(metadata.get("nary_hyperedge_id") or "")
        if card_id and is_card_fact(path):
            grouped[card_id].append(path)

    cards = {}
    for card_id, paths in grouped.items():
        first = paths[0]
        metadata = first.get("metadata", {})
        fact_ids = []
        roles_by_fact = {}
        event_ids = set()
        topic_ids = set()
        for path in paths:
            fact_id = evidence_node_id(path)
            if not fact_id:
                continue
            fact_ids.append(fact_id)
            path_metadata = path.get("metadata", {})
            roles_by_fact.setdefault(fact_id, set()).add(normalize_role(path_metadata.get("nary_role", "evidence")))
            if path_metadata.get("event_node_id"):
                event_ids.add(str(path_metadata["event_node_id"]))
            if path_metadata.get("topic_node_id"):
                topic_ids.add(str(path_metadata["topic_node_id"]))
        fact_ids = list(dict.fromkeys(fact_ids))
        if len(fact_ids) < 1:
            continue
        rank = int(float_meta(metadata, "v3_9_card_rank") or len(cards) + 1)
        card = CardNode(
            card_id=card_id,
            rank=rank,
            relation_type=str(metadata.get("nary_hyperedge_type") or metadata.get("v3_9_cardce_type") or "none"),
            entity=str(metadata.get("v3_9_card_entity") or ""),
            aspect=str(metadata.get("v3_9_card_aspect") or ""),
            summary=str(metadata.get("v3_9_card_summary") or metadata.get("v3_9_cardce_summary") or ""),
            confidence=float_meta(metadata, "nary_hyperedge_confidence"),
            card_ce=float(cardce_scores.get((str(item["question_id"]), card_id), float_meta(metadata, "v3_9_cardce_score"))),
            fact_ids=fact_ids,
            roles_by_fact=roles_by_fact,
            roles=set().union(*roles_by_fact.values()) if roles_by_fact else {"evidence"},
            event_ids=event_ids,
            topic_ids=topic_ids,
        )
        trained_card_point = point_map.get(relation_card_node_id(str(item["question_id"]), card_id))
        card.point = trained_card_point if trained_card_point is not None else card_centroid([point_map[fact_id] for fact_id in fact_ids if fact_id in point_map])
        card.radius_penalty = radius_penalty(card, point_map)
        cards[card_id] = card
    return cards


def infer_role_demand(question: str, cards: dict[str, CardNode]) -> set[str]:
    text = question.lower()
    demand = set()
    if any(token in text for token in ["why", "reason", "because", "cause"]):
        demand.update({"reason", "new_state"})
    if any(token in text for token in ["change", "changed", "switch", "instead", "from ", "to "]):
        demand.update({"old_state", "new_state", "reason"})
    if any(token in text for token in ["prefer", "preference", "favorite", "like", "want"]):
        demand.update({"preference", "constraint"})
    if any(token in text for token in ["when", "date", "time", "before", "after", "later", "earlier"]):
        demand.add("time")
    if any(token in text for token in ["where", "location", "place"]):
        demand.add("context")
    if any(token in text for token in ["plan", "schedule", "deadline", "constraint", "available", "availability", "cannot", "can't"]):
        demand.update({"constraint", "time"})
    if not demand:
        relation_types = {card.relation_type for card in cards.values()}
        if "preference" in relation_types:
            demand.add("preference")
        elif "change" in relation_types:
            demand.update({"old_state", "new_state"})
        elif "temporal" in relation_types:
            demand.add("time")
    return demand or set(DEFAULT_DEMAND)


def beam_search_cards(
    item: dict,
    cards: dict[str, CardNode],
    demand: set[str],
    args,
    *,
    use_hyp: bool,
    lgbm_mu: float,
) -> list[PathState]:
    if not cards:
        return []
    CARD_CONTEXT.clear()
    CARD_CONTEXT.update(cards)
    q_anchor = query_anchor(cards)
    starts = sorted(cards.values(), key=lambda card: card_start_score(card, q_anchor, use_hyp, args), reverse=True)[: args.start_cards]
    beam = [initial_state(card, item, demand, args, use_hyp, lgbm_mu, q_anchor) for card in starts]
    all_states = list(beam)
    for _ in range(1, args.max_cards):
        expanded = []
        for state in beam:
            last = cards[state.card_ids[-1]]
            neighbors = ranked_neighbors(last, cards, state, item, demand, args, use_hyp, lgbm_mu)[: args.neighbors]
            for card in neighbors:
                expanded.append(extend_state(state, card, item, demand, args, use_hyp, lgbm_mu, q_anchor))
        if not expanded:
            break
        beam = sorted(expanded, key=lambda state: state.score, reverse=True)[: args.beam_size]
        all_states.extend(beam)
    return sorted(all_states, key=lambda state: state.score, reverse=True)[: args.beam_size]


def initial_state(card: CardNode, item: dict, demand: set[str], args, use_hyp: bool, lgbm_mu: float, q_anchor: np.ndarray | None) -> PathState:
    facts = select_card_facts(card, item, demand, args, lgbm_mu)
    components = path_components([card], facts, item, demand, args, use_hyp, lgbm_mu, q_anchor)
    return PathState((card.card_id,), sum(components.values()), frozenset(covered_demand_roles([card], demand)), tuple(facts), components)


def extend_state(state: PathState, card: CardNode, item: dict, demand: set[str], args, use_hyp: bool, lgbm_mu: float, q_anchor: np.ndarray | None) -> PathState:
    existing_facts = list(state.facts)
    new_facts = [fact_id for fact_id in select_card_facts(card, item, demand, args, lgbm_mu) if fact_id not in existing_facts]
    card_ids = (*state.card_ids, card.card_id)
    cards = [CARD_CONTEXT[card_id] for card_id in card_ids]
    facts = tuple([*existing_facts, *new_facts])
    components = path_components(cards, list(facts), item, demand, args, use_hyp, lgbm_mu, q_anchor)
    return PathState(card_ids, sum(components.values()), frozenset(covered_demand_roles(cards, demand)), facts, components)


CARD_CONTEXT: dict[str, CardNode] = {}


def path_components(
    cards: list[CardNode],
    facts: list[str],
    item: dict,
    demand: set[str],
    args,
    use_hyp: bool,
    lgbm_mu: float,
    q_anchor: np.ndarray | None,
) -> dict[str, float]:
    role_cov = len(covered_demand_roles(cards, demand)) / max(len(demand), 1)
    rel = sum(card.card_ce + 0.25 * card.confidence for card in cards)
    fact_prior = sum(fact_score(item, fact_id, "base") for fact_id in facts) / max(len(facts), 1)
    chain = sum(link_score(cards[i], cards[i + 1], demand, use_hyp=False) for i in range(len(cards) - 1))
    redundancy = redundancy_penalty(cards, facts)
    cost = len(facts) + 0.5 * len(cards)
    hyp = 0.0
    if use_hyp:
        if q_anchor is not None and cards and cards[0].point is not None:
            hyp -= normalized_distance(q_anchor, cards[0].point)
        for i in range(len(cards) - 1):
            hyp -= normalized_distance(cards[i].point, cards[i + 1].point)
        hyp -= 0.25 * sum(card.radius_penalty for card in cards)
    return {
        "rel": rel,
        "role": args.role_beta * role_cov,
        "chain": args.chain_eta * chain,
        "hyp": args.hyp_gamma * hyp,
        "lgbm_prior": lgbm_mu * fact_prior,
        "redundancy": -args.redundancy_delta * redundancy,
        "cost": -args.cost_lambda * cost,
    }


def paths_from_states(item: dict, cards: dict[str, CardNode], states: list[PathState], topk: int, args, *, lgbm_mu: float) -> list[dict]:
    path_by_fact = {evidence_node_id(path): path for path in item.get("paths", []) if evidence_node_id(path)}
    selected = []
    seen = set()
    for state in states:
        state_cards = [cards[card_id] for card_id in state.card_ids if card_id in cards]
        ranked_facts = sorted(
            state.facts,
            key=lambda fact_id: fact_output_score(fact_id, state_cards, item, lgbm_mu),
            reverse=True,
        )
        for fact_id in ranked_facts:
            if fact_id in seen or fact_id not in path_by_fact:
                continue
            selected.append(mark_path(path_by_fact[fact_id], "v4_1_card_path", state))
            seen.add(fact_id)
            if len(selected) >= topk:
                break
        if len(selected) >= topk:
            break
    if len(selected) < topk:
        for path in sorted(item.get("paths", []), key=lambda path: fact_output_score(evidence_node_id(path), [], item, lgbm_mu), reverse=True):
            fact_id = evidence_node_id(path)
            if fact_id and fact_id not in seen:
                selected.append(mark_path(path, "v4_1_backfill", None))
                seen.add(fact_id)
            if len(selected) >= topk:
                break
    return selected


def select_card_facts(card: CardNode, item: dict, demand: set[str], args, lgbm_mu: float) -> list[str]:
    def score(fact_id: str) -> float:
        roles = card.roles_by_fact.get(fact_id, set())
        role_hit = len(canonical_roles(roles) & demand)
        return 1.0 * role_hit + fact_score(item, fact_id, "cross_encoder") + lgbm_mu * fact_score(item, fact_id, "base")

    return sorted(card.fact_ids, key=score, reverse=True)[: args.max_facts_per_card]


def ranked_neighbors(
    card: CardNode,
    cards: dict[str, CardNode],
    state: PathState,
    item: dict,
    demand: set[str],
    args,
    use_hyp: bool,
    lgbm_mu: float,
) -> list[CardNode]:
    used = set(state.card_ids)
    missing = demand - set(state.covered_roles)

    def score(other: CardNode) -> float:
        role_gain = len(canonical_roles(other.roles) & missing)
        hyp_link = -normalized_distance(card.point, other.point) if use_hyp else 0.0
        return other.card_ce + 0.8 * role_gain + args.chain_eta * link_score(card, other, demand, use_hyp=False) + args.hyp_gamma * hyp_link

    return sorted([other for other in cards.values() if other.card_id not in used], key=score, reverse=True)


def link_score(left: CardNode, right: CardNode, demand: set[str], *, use_hyp: bool) -> float:
    shared_facts = len(set(left.fact_ids) & set(right.fact_ids))
    same_event = 1.0 if left.event_ids & right.event_ids else 0.0
    same_topic = 0.5 if left.topic_ids & right.topic_ids else 0.0
    same_entity = 1.0 if left.entity and left.entity == right.entity else 0.0
    same_aspect = 0.5 if left.aspect and left.aspect == right.aspect else 0.0
    complementary = len((canonical_roles(left.roles) | canonical_roles(right.roles)) & demand) - max(
        len(canonical_roles(left.roles) & demand),
        len(canonical_roles(right.roles) & demand),
    )
    return 0.7 * min(shared_facts, 2) + same_event + same_topic + same_entity + same_aspect + 0.8 * max(complementary, 0)


def card_start_score(card: CardNode, q_anchor: np.ndarray | None, use_hyp: bool, args) -> float:
    score = card.card_ce + 0.25 * card.confidence - 0.03 * card.rank
    if use_hyp and q_anchor is not None and card.point is not None:
        score -= args.hyp_gamma * normalized_distance(q_anchor, card.point)
    return score


def covered_demand_roles(cards: list[CardNode], demand: set[str]) -> set[str]:
    covered = set()
    for card in cards:
        covered.update(canonical_roles(card.roles) & demand)
    return covered


def canonical_roles(roles: set[str]) -> set[str]:
    out = set()
    for role in roles:
        normalized = normalize_role(role)
        for canonical, aliases in ROLE_ALIASES.items():
            if normalized in aliases:
                out.add(canonical)
    return out or {"context"}


def query_anchor(cards: dict[str, CardNode]) -> np.ndarray | None:
    ranked = sorted(
        [card for card in cards.values() if card.point is not None],
        key=lambda card: card.card_ce + 0.25 * card.confidence,
        reverse=True,
    )[:5]
    if not ranked:
        return None
    weights = [max(card.card_ce, 0.0) + 0.25 * max(card.confidence, 0.0) + 1e-3 for card in ranked]
    return card_centroid([card.point for card in ranked if card.point is not None], weights)


def card_centroid(points: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray | None:
    if not points:
        return None
    arr = np.stack(points).astype(np.float64)
    if weights is None:
        mean = arr.mean(axis=0)
    else:
        w = np.asarray(weights, dtype=np.float64)
        w = w / max(float(w.sum()), 1e-12)
        mean = (arr * w[:, None]).sum(axis=0)
    return project_lorentz(mean)


def project_lorentz(vector: np.ndarray) -> np.ndarray:
    spatial_norm_sq = float(np.dot(vector[1:], vector[1:]))
    out = np.asarray(vector, dtype=np.float64).copy()
    out[0] = math.sqrt(max(1.0 + spatial_norm_sq, 1.0))
    return out.astype(np.float32)


def radius_penalty(card: CardNode, point_map: dict[str, np.ndarray]) -> float:
    if card.point is None:
        return 0.0
    fact_radii = [origin_distance(point_map[fact_id]) for fact_id in card.fact_ids if fact_id in point_map]
    if not fact_radii:
        return 0.0
    card_r = origin_distance(card.point)
    return abs(card_r - float(np.mean(fact_radii)))


def origin_distance(point: np.ndarray | None) -> float:
    if point is None:
        return 0.0
    return float(np.arccosh(max(float(point[0]), 1.0 + 1e-6)))


def normalized_distance(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None:
        return 0.0
    d = lorentz_distance(left, right)
    return min(d / 10.0, 2.0)


def lorentz_distance(left: np.ndarray, right: np.ndarray) -> float:
    inner = -float(left[0] * right[0]) + float(np.dot(left[1:], right[1:]))
    return float(np.arccosh(max(-inner, 1.0 + 1e-6)))


def redundancy_penalty(cards: list[CardNode], facts: list[str]) -> float:
    duplicate_facts = len(facts) - len(set(facts))
    event_counts = Counter(event for card in cards for event in card.event_ids)
    topic_counts = Counter(topic for card in cards for topic in card.topic_ids)
    return duplicate_facts + sum(max(count - 1, 0) for count in event_counts.values()) + 0.5 * sum(max(count - 1, 0) for count in topic_counts.values())


def fact_output_score(fact_id: str, cards: list[CardNode], item: dict, lgbm_mu: float) -> float:
    card_bonus = 0.0
    for card in cards:
        if fact_id in card.fact_ids:
            card_bonus = max(card_bonus, card.card_ce + 0.2 * card.confidence)
    return card_bonus + fact_score(item, fact_id, "cross_encoder") + lgbm_mu * fact_score(item, fact_id, "base")


def fact_score(item: dict, fact_id: str, score_name: str) -> float:
    for path in item.get("paths", []):
        if evidence_node_id(path) == fact_id:
            if score_name == "cross_encoder":
                return float(path.get("scores", {}).get("cross_encoder", path.get("score", 0.0)) or 0.0)
            if score_name == "base":
                return float(path.get("scores", {}).get("base", path.get("score", 0.0)) or 0.0)
    return 0.0


def make_item(item: dict, paths: list[dict], method: str, states: list[PathState], demand: set[str]) -> dict:
    copied = dict(item)
    copied["paths"] = paths
    metadata = dict(copied.get("metadata", {}))
    metadata["method"] = method
    metadata["v4_1_role_demand"] = sorted(demand)
    if states:
        metadata["v4_1_best_cards"] = list(states[0].card_ids)
        metadata["v4_1_best_score"] = states[0].score
        metadata["v4_1_best_components"] = states[0].components
    copied["metadata"] = metadata
    return copied


def mark_path(path: dict, source: str, state: PathState | None) -> dict:
    copied = dict(path)
    metadata = dict(copied.get("metadata", {}))
    metadata["v4_1_source"] = source
    if state is not None:
        metadata["v4_1_path_cards"] = list(state.card_ids)
        metadata["v4_1_path_score"] = f"{state.score:.6f}"
        metadata["v4_1_covered_roles"] = sorted(state.covered_roles)
    copied["metadata"] = metadata
    return copied


def path_summary(item: dict, graph, cards: dict[str, CardNode], states: list[PathState], demand: set[str]) -> dict:
    gold = gold_set(item)
    best = states[0] if states else None
    facts = list(best.facts) if best else []
    gold_facts = [fact_id for fact_id in facts if fact_evidence(graph, fact_id) & gold]
    selected_cards = [cards[card_id] for card_id in best.card_ids if card_id in cards] if best else []
    return {
        "question_id": item["question_id"],
        "demand": sorted(demand),
        "best_score": best.score if best else 0.0,
        "card_count": len(selected_cards),
        "fact_count": len(facts),
        "gold_fact_count": len(gold_facts),
        "gold_fact_rate": len(gold_facts) / max(len(facts), 1),
        "covered_roles": sorted(best.covered_roles) if best else [],
        "role_coverage": len(best.covered_roles) / max(len(demand), 1) if best else 0.0,
        "card_types": [card.relation_type for card in selected_cards],
    }


def aggregate_path_summaries(rows: list[dict]) -> dict:
    if not rows:
        return {}
    type_counts = Counter(t for row in rows for t in row.get("card_types", []))
    return {
        "avg_card_count": sum(row["card_count"] for row in rows) / len(rows),
        "avg_fact_count": sum(row["fact_count"] for row in rows) / len(rows),
        "avg_gold_fact_rate": sum(row["gold_fact_rate"] for row in rows) / len(rows),
        "avg_role_coverage": sum(row["role_coverage"] for row in rows) / len(rows),
        "questions_with_path_gold": sum(1 for row in rows if row["gold_fact_count"] > 0),
        "card_type_counts": dict(type_counts),
    }


def compare_items(graph, baseline_items: list[dict], method_items: list[dict], k: int) -> dict:
    base_by_qid = {str(item["question_id"]): item for item in baseline_items}
    hit_fixed = hit_regressed = full_fixed = full_regressed = 0
    examples = []
    for item in method_items:
        base = base_by_qid.get(str(item["question_id"]))
        if base is None:
            continue
        base_eval = evaluate_item(graph, base, k)
        method_eval = evaluate_item(graph, item, k)
        if not base_eval.hit and method_eval.hit:
            hit_fixed += 1
        if base_eval.hit and not method_eval.hit:
            hit_regressed += 1
        if not base_eval.full_cover and method_eval.full_cover:
            full_fixed += 1
        if base_eval.full_cover and not method_eval.full_cover:
            full_regressed += 1
        if len(examples) < 20 and ((not base_eval.full_cover and method_eval.full_cover) or (not base_eval.hit and method_eval.hit)):
            examples.append(
                {
                    "question_id": item["question_id"],
                    "question": item.get("question", ""),
                    "base_hit": base_eval.hit,
                    "method_hit": method_eval.hit,
                    "base_full_cover": base_eval.full_cover,
                    "method_full_cover": method_eval.full_cover,
                    "method_cards": item.get("metadata", {}).get("v4_1_best_cards", []),
                    "method_components": item.get("metadata", {}).get("v4_1_best_components", {}),
                }
            )
    return {
        "hit_fixed": hit_fixed,
        "hit_regressed": hit_regressed,
        "full_cover_fixed": full_fixed,
        "full_cover_regressed": full_regressed,
        "examples": examples,
    }


def load_cv_paths(cv_dir: Path, method: str) -> list[dict]:
    rows = []
    for path in sorted(cv_dir.glob(f"fold_*/*{method}_paths.json")):
        rows.extend(read_json(path))
    return sorted(rows, key=lambda item: item["question_id"])


def is_card_fact(path: dict) -> bool:
    metadata = path.get("metadata", {})
    return str(metadata.get("v3_9_query_card", "")).lower() == "true" or "v3_9_query_card" in str(metadata.get("nary_extractor_type", "")).lower()


def evidence_node_id(path: dict) -> str:
    metadata = path.get("metadata", {})
    if metadata.get("evidence_node_id"):
        return str(metadata["evidence_node_id"])
    node_ids = path.get("node_ids", [])
    return str(node_ids[-1]) if node_ids else ""


def relation_card_node_id(question_id: str, card_id: str) -> str:
    return f"{question_id}:relation_card:{card_id.replace(':', '_')}"


def normalize_role(value: object) -> str:
    return str(value or "evidence").strip().lower().replace("-", "_").replace(" ", "_")


def float_meta(metadata: dict, key: str) -> float:
    try:
        return float(metadata.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def fact_evidence(graph, fact_id: str) -> set[str]:
    node = graph.nodes.get(fact_id)
    if node is None:
        return set()
    out = {normalize_evidence_id(eid) for eid in node.support_ids}
    if not out:
        out.add(normalize_evidence_id(fact_id))
    return out


def gold_set(item: dict) -> set[str]:
    return {normalize_evidence_id(eid) for eid in item.get("gold_evidence", [])}


def render_markdown(payload: dict) -> str:
    lines = ["# V4.1 Hyperbolic Card-Path Search", ""]
    lines.extend(["## Retrieval", "", "| Method | K | Hit | Recall | FullCover | AvgTokens |", "|---|---:|---:|---:|---:|---:|"])
    for method, by_k in payload["retrieval"].items():
        for k, row in by_k.items():
            lines.append(
                f"| {method} | {k} | {row['hit']:.4f} | {row['recall']:.4f} | "
                f"{row['full_cover']:.4f} | {row['avg_tokens']:.1f} |"
            )
    lines.extend(["", "## Path Analysis", "", "| Method | AvgCards | AvgFacts | GoldFactRate | RoleCoverage | QWithPathGold |", "|---|---:|---:|---:|---:|---:|"])
    for method, row in payload["path_analysis"].items():
        lines.append(
            f"| {method} | {row.get('avg_card_count', 0):.2f} | {row.get('avg_fact_count', 0):.2f} | "
            f"{row.get('avg_gold_fact_rate', 0):.4f} | {row.get('avg_role_coverage', 0):.4f} | "
            f"{row.get('questions_with_path_gold', 0)} |"
        )
    if payload["fixes_vs_baseline"]:
        lines.extend(["", "## Fixes vs H0 Baseline", "", "| Method | K | HitFixed | HitRegressed | FullFixed | FullRegressed |", "|---|---:|---:|---:|---:|---:|"])
        for method, by_k in payload["fixes_vs_baseline"].items():
            for k, row in by_k.items():
                lines.append(
                    f"| {method} | {k} | {row['hit_fixed']} | {row['hit_regressed']} | "
                    f"{row['full_cover_fixed']} | {row['full_cover_regressed']} |"
                )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
