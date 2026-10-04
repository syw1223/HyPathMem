from __future__ import annotations

import re
import random
from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np

from hytopomem.memory.hierarchy_builder import (
    cosine,
    extract_entities,
    jaccard,
    parse_turn_order,
    session_from_turn,
)
from hytopomem.memory.node_extractor import content_terms
from hytopomem.memory.schema import Edge, MemoryGraph, Node, NodeType, RelationType


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_QUESTION_START_RE = re.compile(
    r"^(?:what|when|where|who|why|how|did|do|does|is|are|was|were|can|could|would|will|have|has)\b",
    re.IGNORECASE,
)
_INFORMATIVE_PATTERNS = {
    "time": re.compile(
        r"\b(?:today|tonight|yesterday|tomorrow|last|next|ago|recently|currently|"
        r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
        r"january|february|march|april|may|june|july|august|september|october|november|december)\b",
        re.IGNORECASE,
    ),
    "number": re.compile(r"\b\d+(?:st|nd|rd|th)?\b", re.IGNORECASE),
    "state": re.compile(
        r"\b(?:am|is|are|was|were|have|has|had|live|lives|work|works|study|studies|"
        r"like|likes|love|loves|prefer|prefers|want|wants|plan|plans|hope|hopes|"
        r"need|needs|feel|feels|believe|believes|know|knows|own|owns)\b",
        re.IGNORECASE,
    ),
    "event": re.compile(
        r"\b(?:went|go|visited|traveled|travelled|started|finished|joined|attended|"
        r"bought|made|created|painted|played|won|lost|met|adopted|moved|changed|"
        r"received|celebrated|camped|hiked|learned|decided|booked|applied|graduated|"
        r"married|divorced|helped|volunteered|built|wrote|read|watched|cooked|"
        r"having|enjoying)\b",
        re.IGNORECASE,
    ),
    "relation": re.compile(
        r"\b(?:mother|father|mom|dad|parent|parents|sister|brother|husband|wife|"
        r"partner|friend|friends|daughter|son|children|kids|family|mentor|colleague)\b",
        re.IGNORECASE,
    ),
}
_LOW_INFORMATION_PATTERNS = [
    re.compile(r"^(?:thanks|thank you|wow|cool|awesome|great|nice|okay|ok|sure|yeah|yep|aww|haha)[!,. ]*$", re.I),
    re.compile(
        r"^(?:that|it|this) (?:is|'s|sounds|looks) "
        r"(?:great|awesome|cool|nice|amazing|wonderful|impressive|fantastic|inspiring)[!,. ]*$",
        re.I,
    ),
    re.compile(r"^(?:i agree|sounds good|good luck|congratulations|congrats)[!,. ]*$", re.I),
]


class TextEncoder(Protocol):
    model_name_or_path: str

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class SemanticHierarchyConfig:
    event_similarity_threshold: float = 0.52
    topic_similarity_threshold: float = 0.52
    event_max_facts: int = 6
    topic_max_events: int = 8
    event_max_session_gap: int = 1
    max_event_cluster_scan: int = 256
    max_entities: int = 5
    max_keywords: int = 8
    event_summary_chars: int = 260
    topic_summary_chars: int = 360
    include_uncovered_rule_facts: bool = False
    rule_fact_policy: str = "none"
    rule_quality_threshold: float = 0.60
    random_rule_count: int = 944
    random_rule_seed: int = 13

    def __post_init__(self) -> None:
        if self.rule_fact_policy not in {"none", "filtered", "random", "all"}:
            raise ValueError(f"unsupported rule_fact_policy: {self.rule_fact_policy}")
        if self.random_rule_count < 0:
            raise ValueError("random_rule_count must be non-negative")


@dataclass(frozen=True)
class SemanticFact:
    node: Node
    vector: np.ndarray
    terms: set[str]
    entities: set[str]
    session: str
    speaker: str
    order: tuple
    support_raw_ids: tuple[str, ...]


@dataclass
class SemanticCluster:
    members: list[SemanticFact]
    centroid: np.ndarray
    terms: set[str] = field(default_factory=set)
    entities: set[str] = field(default_factory=set)
    sessions: set[str] = field(default_factory=set)
    speakers: set[str] = field(default_factory=set)

    @classmethod
    def from_fact(cls, fact: SemanticFact) -> SemanticCluster:
        return cls(
            members=[fact],
            centroid=fact.vector.copy(),
            terms=set(fact.terms),
            entities=set(fact.entities),
            sessions={fact.session} if fact.session else set(),
            speakers={fact.speaker} if fact.speaker else set(),
        )

    def add(self, fact: SemanticFact) -> None:
        count = len(self.members)
        self.centroid = normalize_vector((self.centroid * count + fact.vector) / (count + 1))
        self.members.append(fact)
        self.terms.update(fact.terms)
        self.entities.update(fact.entities)
        if fact.session:
            self.sessions.add(fact.session)
        if fact.speaker:
            self.speakers.add(fact.speaker)


@dataclass(frozen=True)
class EventView:
    node: Node
    vector: np.ndarray
    terms: set[str]
    entities: set[str]
    sessions: set[str]
    order: tuple
    facts: tuple[SemanticFact, ...]


@dataclass
class EventCluster:
    members: list[EventView]
    centroid: np.ndarray
    terms: set[str] = field(default_factory=set)
    entities: set[str] = field(default_factory=set)
    sessions: set[str] = field(default_factory=set)

    @classmethod
    def from_event(cls, event: EventView) -> EventCluster:
        return cls(
            members=[event],
            centroid=event.vector.copy(),
            terms=set(event.terms),
            entities=set(event.entities),
            sessions=set(event.sessions),
        )

    def add(self, event: EventView) -> None:
        count = len(self.members)
        self.centroid = normalize_vector((self.centroid * count + event.vector) / (count + 1))
        self.members.append(event)
        self.terms.update(event.terms)
        self.entities.update(event.entities)
        self.sessions.update(event.sessions)


@dataclass(frozen=True)
class RuleStatement:
    text: str
    score: float
    signals: tuple[str, ...]
    source_fact_id: str


class SemanticHierarchyBuilder:
    """Build an event-first FACT -> EVENT -> TOPIC hierarchy.

    Observation FACTs are the preferred semantic units. Rule FACTs whose RAW
    supports are already covered by observations are retained only as lexical
    aliases, so bottom-up retrieval can still enter the semantic hierarchy.
    """

    def __init__(
        self,
        encoder: TextEncoder,
        config: SemanticHierarchyConfig | None = None,
    ):
        self.encoder = encoder
        self.config = config or SemanticHierarchyConfig()

    def build(self, graph: MemoryGraph, graph_id: str | None = None) -> MemoryGraph:
        output = strip_generated_hierarchy(graph)
        output.graph_id = graph_id or f"{graph.graph_id}_semantic_hierarchy_v3"
        canonical_by_conv, aliases_by_conv, rule_stats = self._canonical_facts(output)
        known_entities_by_conv = self._known_entities(output)
        all_facts = [fact for facts in canonical_by_conv.values() for fact in facts]
        embeddings = self.encoder.encode([fact.text for fact in all_facts])
        vector_by_id = {
            fact.node_id: normalize_vector(vector)
            for fact, vector in zip(all_facts, embeddings)
        }

        event_count = 0
        topic_count = 0
        hierarchy_edge_count = 0
        alias_edge_count = 0
        canonical_count = 0
        canonical_source_counts: dict[str, int] = {}
        for conversation_id in sorted(canonical_by_conv):
            nodes = canonical_by_conv[conversation_id]
            views = [
                self._fact_view(
                    node,
                    vector_by_id[node.node_id],
                    known_entities_by_conv.get(conversation_id, set()),
                )
                for node in sorted(nodes, key=self._node_order)
            ]
            canonical_count += len(views)
            for view in views:
                canonical_source_counts[view.node.source] = canonical_source_counts.get(view.node.source, 0) + 1

            event_segments = self._event_clusters(views)
            event_views: list[EventView] = []
            raw_to_event: dict[str, str] = {}
            for event_index, segment in enumerate(event_segments, start=1):
                event_count += 1
                event_id = f"{conversation_id}:eventv3:{event_index:04d}"
                event_node = self._event_node(event_id, conversation_id, segment)
                output.add_node(event_node)
                event_view = self._event_view(event_node, segment)
                event_views.append(event_view)
                confidence = self._cluster_confidence([fact.vector for fact in segment])
                for fact in segment:
                    output.add_edge(
                        Edge(
                            src=fact.node.node_id,
                            dst=event_id,
                            relation=RelationType.IS_SPECIFIC_OF,
                            confidence=confidence,
                            metadata={
                                "hierarchy_v3": "fact_event",
                                "builder": "event_first_semantic_v1",
                                "canonical_fact": True,
                            },
                        )
                    )
                    hierarchy_edge_count += 1
                    for raw_id in fact.support_raw_ids:
                        raw_to_event.setdefault(raw_id, event_id)

            topic_segments = self._topic_clusters(event_views)
            for topic_index, topic_events in enumerate(topic_segments, start=1):
                topic_count += 1
                topic_id = f"{conversation_id}:topicv3:{topic_index:04d}"
                topic_node = self._topic_node(topic_id, conversation_id, topic_events)
                output.add_node(topic_node)
                confidence = self._cluster_confidence([event.vector for event in topic_events])
                for event in topic_events:
                    output.add_edge(
                        Edge(
                            src=event.node.node_id,
                            dst=topic_id,
                            relation=RelationType.IS_SPECIFIC_OF,
                            confidence=confidence,
                            metadata={
                                "hierarchy_v3": "event_topic",
                                "builder": "event_first_semantic_v1",
                            },
                        )
                    )
                    hierarchy_edge_count += 1
                    event_node = output.nodes[event.node.node_id]
                    event_node.metadata["topic_id"] = topic_id

            for alias in aliases_by_conv.get(conversation_id, []):
                event_id = first_supported_event(alias, raw_to_event)
                if event_id is None:
                    continue
                output.add_edge(
                    Edge(
                        src=alias.node_id,
                        dst=event_id,
                        relation=RelationType.IS_SPECIFIC_OF,
                        confidence=0.90,
                        metadata={
                            "hierarchy_v3": "lexical_alias_event",
                            "builder": "event_first_semantic_v1",
                            "canonical_fact": False,
                        },
                    )
                )
                alias_edge_count += 1

        metadata = dict(output.metadata)
        total_rule_facts = sum(
            node.source == "rule_extracted"
            for node in output.iter_nodes(NodeType.FACT)
        )
        canonical_rule_facts = canonical_source_counts.get("rule_extracted", 0)
        metadata["hierarchy_v3"] = {
            "builder": "event_first_semantic_v1",
            "encoder": self.encoder.model_name_or_path,
            "config": self.config.__dict__,
            "source_graph_id": graph.graph_id,
            "canonical_fact_nodes": canonical_count,
            "canonical_source_counts": canonical_source_counts,
            "lexical_alias_edges": alias_edge_count,
            "rule_filter": rule_stats,
            "unclustered_rule_facts": max(
                0,
                total_rule_facts - canonical_rule_facts - alias_edge_count,
            ),
            "event_nodes": event_count,
            "topic_nodes": topic_count,
            "hierarchy_edges": hierarchy_edge_count,
        }
        output.metadata = metadata
        return output

    def _canonical_facts(
        self,
        graph: MemoryGraph,
    ) -> tuple[dict[str, list[Node]], dict[str, list[Node]], dict]:
        by_conv: dict[str, list[Node]] = {}
        for node in graph.iter_nodes(NodeType.FACT):
            by_conv.setdefault(conversation_id(node.node_id), []).append(node)

        canonical: dict[str, list[Node]] = {}
        aliases: dict[str, list[Node]] = {}
        effective_policy = "all" if self.config.include_uncovered_rule_facts else self.config.rule_fact_policy
        rule_groups: dict[str, tuple[list[Node], list[Node], list[Node]]] = {}
        all_fallback_rules: list[Node] = []
        for conv_id, facts in by_conv.items():
            observations = [
                fact
                for fact in facts
                if fact.source in {"locomo_observation", "sentence_fact"}
            ]
            rules = [fact for fact in facts if fact.source == "rule_extracted"]
            covered_raw_ids = {
                raw_id
                for fact in observations
                for raw_id in support_raw_ids(fact)
            }
            fallback_rules = [
                fact
                for fact in rules
                if not support_raw_ids(fact) or not set(support_raw_ids(fact)).issubset(covered_raw_ids)
            ]
            covered_rules = [fact for fact in rules if fact not in fallback_rules]
            rule_groups[conv_id] = (observations, fallback_rules, covered_rules)
            all_fallback_rules.extend(fallback_rules)

        random_rule_ids: set[str] = set()
        if effective_policy == "random":
            rng = random.Random(self.config.random_rule_seed)
            sample_size = min(self.config.random_rule_count, len(all_fallback_rules))
            random_rule_ids = {
                fact.node_id
                for fact in rng.sample(sorted(all_fallback_rules, key=lambda item: item.node_id), sample_size)
            }
        stats = {
            "policy": effective_policy,
            "threshold": self.config.rule_quality_threshold,
            "random_rule_count": self.config.random_rule_count,
            "random_rule_seed": self.config.random_rule_seed,
            "covered_rule_aliases": 0,
            "selected_uncovered_rules": 0,
            "rejected_uncovered_rules": 0,
            "selected_signal_counts": {},
            "selected_score_mean": 0.0,
        }
        selected_scores: list[float] = []
        selected_signal_counts: dict[str, int] = {}
        for conv_id, (observations, fallback_rules, covered_rules) in rule_groups.items():
            stats["covered_rule_aliases"] += len(covered_rules)
            canonical_rules: list[Node] = []
            selected_aliases: list[Node] = []
            if effective_policy == "all":
                canonical_rules = fallback_rules
                stats["selected_uncovered_rules"] += len(fallback_rules)
            elif effective_policy == "random":
                canonical_rules = [fact for fact in fallback_rules if fact.node_id in random_rule_ids]
                stats["selected_uncovered_rules"] += len(canonical_rules)
                stats["rejected_uncovered_rules"] += len(fallback_rules) - len(canonical_rules)
            elif effective_policy == "filtered":
                for rule in fallback_rules:
                    statement = extract_rule_statement(rule)
                    if statement is None or statement.score < self.config.rule_quality_threshold:
                        stats["rejected_uncovered_rules"] += 1
                        continue
                    derived = rule_statement_node(rule, statement)
                    graph.add_node(derived)
                    for raw_id in support_raw_ids(derived):
                        if raw_id in graph.nodes:
                            graph.add_edge(
                                Edge(
                                    src=derived.node_id,
                                    dst=raw_id,
                                    relation=RelationType.SUPPORTS,
                                    confidence=0.98,
                                    metadata={
                                        "builder": "query_independent_rule_filter_v1",
                                        "derived_from_fact_id": rule.node_id,
                                    },
                                )
                            )
                    canonical_rules.append(derived)
                    selected_aliases.append(rule)
                    stats["selected_uncovered_rules"] += 1
                    selected_scores.append(statement.score)
                    for signal in statement.signals:
                        selected_signal_counts[signal] = selected_signal_counts.get(signal, 0) + 1
            else:
                stats["rejected_uncovered_rules"] += len(fallback_rules)
            canonical[conv_id] = observations + canonical_rules
            aliases[conv_id] = covered_rules + selected_aliases
        stats["selected_signal_counts"] = selected_signal_counts
        stats["selected_score_mean"] = float(np.mean(selected_scores)) if selected_scores else 0.0
        return canonical, aliases, stats

    def _known_entities(self, graph: MemoryGraph) -> dict[str, set[str]]:
        output: dict[str, set[str]] = {}
        for node in graph.nodes.values():
            if node.type not in {NodeType.RAW, NodeType.FACT}:
                continue
            speaker = str(node.metadata.get("speaker") or "").strip()
            if speaker:
                output.setdefault(conversation_id(node.node_id), set()).add(speaker)
        return output

    def _fact_view(
        self,
        node: Node,
        vector: np.ndarray,
        known_entities: set[str],
    ) -> SemanticFact:
        speaker = str(node.metadata.get("speaker") or "")
        entities = {
            entity
            for entity in extract_entities(node.text)
            if entity in known_entities
        }
        if speaker:
            entities.add(speaker)
        return SemanticFact(
            node=node,
            vector=vector,
            terms=set(content_terms(node.text)),
            entities=entities,
            session=str(
                node.metadata.get("session")
                or node.metadata.get("session_id")
                or session_from_turn(node.metadata.get("turn_id"))
            ),
            speaker=speaker,
            order=self._node_order(node),
            support_raw_ids=tuple(support_raw_ids(node)),
        )

    def _event_clusters(self, facts: Sequence[SemanticFact]) -> list[list[SemanticFact]]:
        clusters: list[SemanticCluster] = []
        for fact in facts:
            best_index = -1
            best_score = -1.0
            scan_start = max(0, len(clusters) - self.config.max_event_cluster_scan)
            for index in range(scan_start, len(clusters)):
                cluster = clusters[index]
                if len(cluster.members) >= self.config.event_max_facts:
                    continue
                gap = session_gap(fact.session, cluster.sessions)
                if session_span_with(fact.session, cluster.sessions) > self.config.event_max_session_gap:
                    continue
                score = self._fact_event_similarity(fact, cluster)
                if gap > 0:
                    score -= 0.06 * gap
                if score > best_score:
                    best_index = index
                    best_score = score
            if best_index >= 0 and best_score >= self.config.event_similarity_threshold:
                clusters[best_index].add(fact)
            else:
                clusters.append(SemanticCluster.from_fact(fact))
        return [sorted(cluster.members, key=lambda fact: fact.order) for cluster in clusters]

    def _topic_clusters(self, events: Sequence[EventView]) -> list[list[EventView]]:
        clusters: list[EventCluster] = []
        for event in sorted(events, key=lambda item: item.order):
            best_index = -1
            best_score = -1.0
            for index, cluster in enumerate(clusters):
                if len(cluster.members) >= self.config.topic_max_events:
                    continue
                score = self._event_topic_similarity(event, cluster)
                if score > best_score:
                    best_index = index
                    best_score = score
            if best_index >= 0 and best_score >= self.config.topic_similarity_threshold:
                clusters[best_index].add(event)
            else:
                clusters.append(EventCluster.from_event(event))
        return [sorted(cluster.members, key=lambda event: event.order) for cluster in clusters]

    def _fact_event_similarity(self, fact: SemanticFact, cluster: SemanticCluster) -> float:
        return (
            0.60 * cosine(fact.vector, cluster.centroid)
            + 0.15 * jaccard(fact.entities, cluster.entities)
            + 0.15 * jaccard(fact.terms, cluster.terms)
            + 0.05 * session_similarity(fact.session, cluster.sessions)
            + 0.05 * float(bool(fact.speaker and fact.speaker in cluster.speakers))
        )

    def _event_topic_similarity(self, event: EventView, cluster: EventCluster) -> float:
        return (
            0.65 * cosine(event.vector, cluster.centroid)
            + 0.20 * jaccard(event.entities, cluster.entities)
            + 0.10 * jaccard(event.terms, cluster.terms)
            + 0.05 * session_similarity_set(event.sessions, cluster.sessions)
        )

    def _event_node(
        self,
        node_id: str,
        conversation_id: str,
        facts: Sequence[SemanticFact],
    ) -> Node:
        centroid = mean_vector([fact.vector for fact in facts])
        representative = max(facts, key=lambda fact: cosine(fact.vector, centroid))
        entities = ranked_values(facts, "entities", self.config.max_entities)
        keywords = ranked_values(facts, "terms", self.config.max_keywords)
        summary = event_summary(representative.node.text, entities, self.config.event_summary_chars)
        coherence = cluster_coherence([fact.vector for fact in facts])
        return Node(
            node_id=node_id,
            type=NodeType.EVENT,
            text=summary,
            source="semantic_event_cluster_v1",
            confidence=self._cluster_confidence([fact.vector for fact in facts]),
            support_ids=[fact.node.node_id for fact in facts],
            metadata={
                "conversation_id": conversation_id,
                "fact_ids": [fact.node.node_id for fact in facts],
                "support_raw_ids": sorted({raw_id for fact in facts for raw_id in fact.support_raw_ids}),
                "session_ids": sorted(
                    {fact.session for fact in facts if fact.session},
                    key=semantic_session_order,
                ),
                "entities": entities,
                "keywords": keywords,
                "coherence": coherence,
                "representative_fact_id": representative.node.node_id,
                "hierarchy_v3": "event",
                "label_type": "template",
            },
        )

    def _event_view(self, node: Node, facts: Sequence[SemanticFact]) -> EventView:
        return EventView(
            node=node,
            vector=mean_vector([fact.vector for fact in facts]),
            terms=set().union(*(fact.terms for fact in facts)),
            entities=set().union(*(fact.entities for fact in facts)),
            sessions={fact.session for fact in facts if fact.session},
            order=min(fact.order for fact in facts),
            facts=tuple(facts),
        )

    def _topic_node(
        self,
        node_id: str,
        conversation_id: str,
        events: Sequence[EventView],
    ) -> Node:
        centroid = mean_vector([event.vector for event in events])
        representatives = sorted(
            events,
            key=lambda event: cosine(event.vector, centroid),
            reverse=True,
        )[:2]
        entities = ranked_event_values(events, "entities", self.config.max_entities)
        keywords = ranked_event_values(events, "terms", self.config.max_keywords)
        summary = topic_summary(
            [event.node.text for event in representatives],
            entities,
            keywords,
            self.config.topic_summary_chars,
        )
        coherence = cluster_coherence([event.vector for event in events])
        return Node(
            node_id=node_id,
            type=NodeType.TOPIC,
            text=summary,
            source="semantic_event_community_v1",
            confidence=self._cluster_confidence([event.vector for event in events]),
            support_ids=[event.node.node_id for event in events],
            metadata={
                "conversation_id": conversation_id,
                "event_ids": [event.node.node_id for event in events],
                "fact_ids": [fact.node.node_id for event in events for fact in event.facts],
                "session_ids": sorted(
                    {session for event in events for session in event.sessions},
                    key=semantic_session_order,
                ),
                "entities": entities,
                "keywords": keywords,
                "coherence": coherence,
                "hierarchy_v3": "topic",
                "label_type": "template",
            },
        )

    def _cluster_confidence(self, vectors: Sequence[np.ndarray]) -> float:
        return min(0.96, 0.58 + 0.36 * cluster_coherence(vectors))

    def _node_order(self, node: Node) -> tuple:
        turn_id = str(node.metadata.get("turn_id") or "")
        if turn_id:
            return parse_turn_order(turn_id)
        support_turn_ids = node.metadata.get("support_turn_ids") or []
        if support_turn_ids:
            return min(parse_turn_order(str(item)) for item in support_turn_ids)
        session = str(node.metadata.get("session") or "")
        suffix = node.node_id.rsplit(":", 1)[-1]
        try:
            number = int(suffix)
        except ValueError:
            number = 999999
        return (semantic_session_order(session), 1, number, node.node_id)


def strip_generated_hierarchy(graph: MemoryGraph) -> MemoryGraph:
    output = graph.model_copy(deep=True)
    generated_ids = {
        node_id
        for node_id, node in output.nodes.items()
        if node.type in {NodeType.EVENT, NodeType.TOPIC}
        or "hierarchy_v2" in node.metadata
        or "hierarchy_v3" in node.metadata
    }
    output.nodes = {
        node_id: node
        for node_id, node in output.nodes.items()
        if node_id not in generated_ids
    }
    output.edges = [
        edge
        for edge in output.edges
        if edge.src not in generated_ids
        and edge.dst not in generated_ids
        and "hierarchy_v2" not in edge.metadata
        and "hierarchy_v3" not in edge.metadata
    ]
    metadata = dict(output.metadata)
    metadata.pop("hierarchy_v2", None)
    metadata.pop("hierarchy_v3", None)
    output.metadata = metadata
    return output


def support_raw_ids(node: Node) -> list[str]:
    values = node.metadata.get("support_raw_ids") or node.support_ids
    conv_id = conversation_id(node.node_id)
    normalized = []
    for item in values:
        value = str(item).strip()
        if not value:
            continue
        if ":raw:" in value:
            normalized.append(value)
        elif ":fact:" not in value:
            normalized.append(f"{conv_id}:raw:{value}")
    return normalized


def extract_rule_statement(node: Node) -> RuleStatement | None:
    speaker = str(node.metadata.get("speaker") or "").strip()
    body = node.text.strip()
    said_prefix = f"{speaker} said:" if speaker else ""
    if said_prefix and body.lower().startswith(said_prefix.lower()):
        body = body[len(said_prefix) :].strip()
    candidates = []
    for sentence in _SENTENCE_SPLIT_RE.split(body):
        cleaned = clean_rule_clause(sentence)
        if not cleaned:
            continue
        score, signals = score_rule_clause(cleaned)
        if score <= 0.0:
            continue
        candidates.append((score, cleaned, signals))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[0]
    text = normalize_rule_subject(selected[1], speaker)
    return RuleStatement(
        text=text,
        score=selected[0],
        signals=selected[2],
        source_fact_id=node.node_id,
    )


def clean_rule_clause(text: str) -> str:
    value = " ".join(text.strip().split())
    if not value:
        return ""
    value = re.sub(
        r"^(?:thanks(?:\s+\w+)?|thank you|wow|cool|awesome|great|nice|okay|ok|sure|yeah|yep|aww|haha)"
        r"\s*[,!.-]*\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"^(?:by the way|btw|actually|well)\s*[,!.-]*\s*", "", value, flags=re.IGNORECASE)
    value = value.strip()
    if not value or value.endswith("?") or _QUESTION_START_RE.match(value):
        return ""
    if any(pattern.fullmatch(value) for pattern in _LOW_INFORMATION_PATTERNS):
        return ""
    return value


def score_rule_clause(text: str) -> tuple[float, tuple[str, ...]]:
    terms = content_terms(text)
    words = text.split()
    if len(words) < 5 or len(terms) < 3:
        return 0.0, ()
    first_person = bool(re.search(r"\b(?:i|i'm|i've|i'd|me|my|mine|we|we're|we've|us|our|ours)\b", text, re.I))
    second_person = bool(re.search(r"\b(?:you|your|yours|you're|you've)\b", text, re.I))
    if second_person and not first_person:
        return 0.0, ()
    signals = [name for name, pattern in _INFORMATIVE_PATTERNS.items() if pattern.search(text)]
    has_fact_predicate = bool({"state", "event"} & set(signals))
    if not has_fact_predicate:
        return 0.0, ()
    score = 0.20
    score += min(0.20, 0.025 * len(terms))
    score += min(0.35, 0.10 * len(signals))
    if 7 <= len(words) <= 45:
        score += 0.10
    if first_person:
        score += 0.08
    if re.search(r"\b(?:because|so that|which|who|where|when)\b", text, re.IGNORECASE):
        score += 0.04
    if len(words) > 65:
        score -= 0.12
    return max(0.0, min(1.0, score)), tuple(sorted(signals))


def normalize_rule_subject(text: str, speaker: str) -> str:
    value = text.strip()
    if not speaker:
        return ensure_sentence(value)
    replacements = [
        (r"^I'm\b", f"{speaker} is"),
        (r"^I am\b", f"{speaker} is"),
        (r"^I've\b", f"{speaker} has"),
        (r"^I have\b", f"{speaker} has"),
        (r"^I'd\b", f"{speaker} would"),
        (r"^I'll\b", f"{speaker} will"),
        (r"^I\b", speaker),
        (r"^My\b", f"{speaker}'s"),
        (r"^We're\b", f"{speaker} and others are"),
        (r"^We've\b", f"{speaker} and others have"),
        (r"^We\b", f"{speaker} and others"),
        (r"^Our\b", f"{speaker} and others'"),
    ]
    for pattern, replacement in replacements:
        if re.search(pattern, value, re.IGNORECASE):
            value = re.sub(pattern, replacement, value, count=1, flags=re.IGNORECASE)
            break
    value = re.sub(r"\bI'm\b", f"{speaker} is", value, flags=re.IGNORECASE)
    value = re.sub(r"\bI've\b", f"{speaker} has", value, flags=re.IGNORECASE)
    value = re.sub(r"\bI'd\b", f"{speaker} would", value, flags=re.IGNORECASE)
    value = re.sub(r"\bI'll\b", f"{speaker} will", value, flags=re.IGNORECASE)
    value = re.sub(r"\bI\b", speaker, value, flags=re.IGNORECASE)
    value = re.sub(r"\bme\b", speaker, value, flags=re.IGNORECASE)
    value = re.sub(r"\bmy\b", f"{speaker}'s", value, flags=re.IGNORECASE)
    value = re.sub(r"\bmine\b", f"{speaker}'s", value, flags=re.IGNORECASE)
    value = re.sub(r"\bWe're\b", f"{speaker} and others are", value, flags=re.IGNORECASE)
    value = re.sub(r"\bWe've\b", f"{speaker} and others have", value, flags=re.IGNORECASE)
    value = re.sub(r"\bWe\b", f"{speaker} and others", value, flags=re.IGNORECASE)
    value = re.sub(r"\bus\b", f"{speaker} and others", value, flags=re.IGNORECASE)
    value = re.sub(r"\bour\b", f"{speaker} and others'", value, flags=re.IGNORECASE)
    if value == text.strip() and not value.lower().startswith(speaker.lower()):
        value = f"{speaker} stated that {value[0].lower() + value[1:] if value else value}"
    return ensure_sentence(value)


def ensure_sentence(text: str) -> str:
    value = text.strip()
    if not value:
        return value
    return value if value[-1] in ".!?" else value + "."


def rule_statement_node(rule: Node, statement: RuleStatement) -> Node:
    conv_id = conversation_id(rule.node_id)
    suffix = rule.node_id.split(":fact:", 1)[-1]
    return Node(
        node_id=f"{conv_id}:fact:rule_sem:{suffix}",
        type=NodeType.FACT,
        text=statement.text,
        time=rule.time,
        source="filtered_rule_statement",
        status=rule.status,
        confidence=min(0.88, 0.62 + 0.28 * statement.score),
        support_ids=support_raw_ids(rule),
        metadata={
            **dict(rule.metadata),
            "source_fact_id": rule.node_id,
            "support_raw_ids": support_raw_ids(rule),
            "rule_quality_score": statement.score,
            "rule_quality_signals": list(statement.signals),
            "canonicalization": "query_independent_rule_filter_v1",
        },
    )


def first_supported_event(node: Node, raw_to_event: dict[str, str]) -> str | None:
    for raw_id in support_raw_ids(node):
        if raw_id in raw_to_event:
            return raw_to_event[raw_id]
    return None


def conversation_id(node_id: str) -> str:
    for marker in (":raw:", ":fact_sent:", ":fact:", ":anchor:", ":event", ":topic", ":episode"):
        if marker in node_id:
            return node_id.split(marker, 1)[0]
    return node_id.split(":", 1)[0]


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 1e-12 else value


def mean_vector(vectors: Sequence[np.ndarray]) -> np.ndarray:
    if not vectors:
        raise ValueError("cannot average an empty vector sequence")
    return normalize_vector(np.mean(np.asarray(vectors, dtype=np.float32), axis=0))


def cluster_coherence(vectors: Sequence[np.ndarray]) -> float:
    if len(vectors) < 2:
        return 0.5
    centroid = mean_vector(vectors)
    values = [cosine(vector, centroid) for vector in vectors]
    return max(0.0, min(1.0, float(np.mean(values))))


def session_number(session: str) -> int | None:
    match = re.search(r"(\d+)", session or "")
    return int(match.group(1)) if match else None


def semantic_session_order(session: str) -> str:
    number = session_number(session)
    return f"{number:06d}:{session}" if number is not None else f"999999:{session}"


def session_gap(session: str, cluster_sessions: set[str]) -> int:
    if not session or not cluster_sessions:
        return 0
    if session in cluster_sessions:
        return 0
    number = session_number(session)
    other_numbers = [session_number(item) for item in cluster_sessions]
    valid = [item for item in other_numbers if item is not None]
    if number is None or not valid:
        return 2
    return min(abs(number - item) for item in valid)


def session_span_with(session: str, cluster_sessions: set[str]) -> int:
    numbers = [session_number(item) for item in cluster_sessions]
    if session:
        numbers.append(session_number(session))
    valid = [item for item in numbers if item is not None]
    return max(valid) - min(valid) if valid else 0


def session_similarity(session: str, cluster_sessions: set[str]) -> float:
    gap = session_gap(session, cluster_sessions)
    if gap == 0:
        return 1.0
    if gap == 1:
        return 0.5
    return 0.0


def session_similarity_set(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    if left & right:
        return 1.0
    gaps = [session_gap(item, right) for item in left]
    return 0.5 if gaps and min(gaps) == 1 else 0.0


def ranked_values(items: Sequence[SemanticFact], attribute: str, limit: int) -> list[str]:
    counts: dict[str, float] = {}
    display: dict[str, str] = {}
    for item in items:
        for value in getattr(item, attribute):
            key = value.lower()
            counts[key] = counts.get(key, 0.0) + 1.0
            display.setdefault(key, value)
    ranked = sorted(counts, key=lambda key: (-counts[key], key))
    return [display[key] for key in ranked[:limit]]


def ranked_event_values(items: Sequence[EventView], attribute: str, limit: int) -> list[str]:
    counts: dict[str, float] = {}
    display: dict[str, str] = {}
    for item in items:
        for value in getattr(item, attribute):
            key = value.lower()
            counts[key] = counts.get(key, 0.0) + 1.0
            display.setdefault(key, value)
    ranked = sorted(counts, key=lambda key: (-counts[key], key))
    return [display[key] for key in ranked[:limit]]


def event_summary(representative_text: str, entities: Sequence[str], max_chars: int) -> str:
    prefix = f"Event involving {', '.join(entities[:3])}: " if entities else "Event: "
    text = representative_text.strip()
    if text.lower().startswith("event:"):
        text = text.split(":", 1)[1].strip()
    return truncate_text(prefix + text, max_chars)


def topic_summary(
    event_texts: Sequence[str],
    entities: Sequence[str],
    keywords: Sequence[str],
    max_chars: int,
) -> str:
    subjects = list(entities[:3]) or list(keywords[:4])
    prefix = f"Topic about {', '.join(subjects)}. " if subjects else "Topic. "
    details = "; ".join(strip_label(text, "Event") for text in event_texts)
    return truncate_text(prefix + "Includes " + details, max_chars)


def strip_label(text: str, label: str) -> str:
    prefix = f"{label.lower()}:"
    value = text.strip()
    return value.split(":", 1)[1].strip() if value.lower().startswith(prefix) else value


def truncate_text(text: str, max_chars: int) -> str:
    value = " ".join(text.split())
    if len(value) <= max_chars:
        return value
    shortened = value[: max(1, max_chars - 1)].rsplit(" ", 1)[0]
    return shortened.rstrip(" ,;:.") + "."
