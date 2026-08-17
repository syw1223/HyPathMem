from __future__ import annotations

from enum import StrEnum
from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class NodeType(StrEnum):
    RAW = "RAW"
    FACT = "FACT"
    ANCHOR = "ANCHOR"
    EVENT = "EVENT"
    TOPIC = "TOPIC"
    SESSION = "SESSION"
    CONVERSATION = "CONVERSATION"
    ENTITY = "ENTITY"
    ENTITY_STATE = "ENTITY_STATE"
    EVIDENCE_PACK = "EVIDENCE_PACK"
    RELATION_CARD = "RELATION_CARD"


class NodeStatus(StrEnum):
    ACTIVE = "active"
    OUTDATED = "outdated"
    EXCEPTION = "exception"
    DISPUTED = "disputed"


class RelationType(StrEnum):
    IS_SPECIFIC_OF = "IS_SPECIFIC_OF"
    SUPPORTS = "SUPPORTS"
    UPDATES = "UPDATES"
    EXCEPTION_OF = "EXCEPTION_OF"
    CONFLICTS_WITH = "CONFLICTS_WITH"
    MEMBER_OF = "MEMBER_OF"
    FILLS_ROLE = "FILLS_ROLE"


PARTIAL_ORDER_RELATIONS = {RelationType.IS_SPECIFIC_OF}


class Node(BaseModel):
    node_id: str
    type: NodeType
    text: str
    time: Optional[str] = None
    source: str = "unknown"
    status: NodeStatus = NodeStatus.ACTIVE
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    support_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def non_empty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("node text cannot be empty")
        return value


class Edge(BaseModel):
    edge_id: Optional[str] = None
    src: str
    dst: str
    relation: RelationType
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def fill_edge_id(self) -> "Edge":
        if self.edge_id is None:
            self.edge_id = f"{self.src}->{self.relation}->{self.dst}"
        return self

    @property
    def is_partial_order(self) -> bool:
        return self.relation in PARTIAL_ORDER_RELATIONS


class EvidencePath(BaseModel):
    query_id: str
    anchor_id: Optional[str] = None
    node_ids: List[str] = Field(default_factory=list)
    edge_ids: List[str] = Field(default_factory=list)
    score: float = 0.0
    scores: Dict[str, float] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MemoryGraph(BaseModel):
    graph_id: str
    nodes: Dict[str, Node] = Field(default_factory=dict)
    edges: List[Edge] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def add_node(self, node: Node) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, edge: Edge) -> None:
        if edge.src not in self.nodes:
            raise KeyError(f"missing edge source node: {edge.src}")
        if edge.dst not in self.nodes:
            raise KeyError(f"missing edge destination node: {edge.dst}")
        self.edges.append(edge)

    def iter_nodes(self, node_type: Optional[NodeType] = None) -> Iterable[Node]:
        for node in self.nodes.values():
            if node_type is None or node.type == node_type:
                yield node

    def outgoing(self, node_id: str, relation: Optional[RelationType] = None) -> List[Edge]:
        return [
            edge
            for edge in self.edges
            if edge.src == node_id and (relation is None or edge.relation == relation)
        ]

    def incoming(self, node_id: str, relation: Optional[RelationType] = None) -> List[Edge]:
        return [
            edge
            for edge in self.edges
            if edge.dst == node_id and (relation is None or edge.relation == relation)
        ]

    def partial_order_edges(self) -> List[Edge]:
        return [edge for edge in self.edges if edge.is_partial_order]
