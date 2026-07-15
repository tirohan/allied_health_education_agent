from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class EntityType(StrEnum):
    TOPIC = "Topic"
    COMPETENCY = "Competency"
    DISCIPLINE = "Discipline"
    PAPER = "Paper"
    RESOURCE = "Resource"
    PROGRAM = "Program"
    INSTITUTION = "Institution"
    COUNTY = "County"
    SHORTAGE_AREA = "ShortageArea"
    SIMULATION_CASE = "SimulationCase"
    AUTHOR = "Author"


class RelationType(StrEnum):
    SUPPORTS = "SUPPORTS"
    ADDRESSES = "ADDRESSES"
    OFFERED_AT = "OFFERED_AT"
    ACCREDITED_BY = "ACCREDITED_BY"
    LOCATED_IN = "LOCATED_IN"
    SHORTAGE_FOR = "SHORTAGE_FOR"
    MAPPED_TO = "MAPPED_TO"
    AUTHORED_BY = "AUTHORED_BY"
    TRAINED_FOR = "TRAINED_FOR"
    RELEVANT_TO = "RELEVANT_TO"


class VerificationStatus(StrEnum):
    CONFIRMED = "CONFIRMED"
    INFERRED = "INFERRED"
    UNVERIFIED = "UNVERIFIED"
    REFUTED = "REFUTED"


class RetrievalCollection(StrEnum):
    PAPERS = "papers"
    RESOURCES = "resources"
    PROGRAMS = "programs"
    COMMUNITIES = "communities"
    SIMULATION_CASES = "simulation_cases"


class Citation(BaseModel):
    source_table: str
    source_id: str
    label: str
    url: str | None = None
    doi: str | None = None
    evidence_snippet: str | None = None


class RetrievedDoc(BaseModel):
    id: str
    collection: RetrievalCollection
    source_table: str
    source_id: str
    title: str
    text: str
    score: float = Field(ge=0.0)
    vector_rank: int | None = Field(default=None, ge=1)
    sql_rank: int | None = Field(default=None, ge=1)
    payload: dict[str, Any] = Field(default_factory=dict)


class Entity(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    entity_id: str
    entity_type: EntityType
    label: str
    summary: str = ""
    source_table: str
    source_id: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_doc_ids: list[str] = Field(default_factory=list)


class Relation(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    relation_id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: RelationType
    evidence_text: str = ""
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_doc_ids: list[str] = Field(default_factory=list)


class VerificationResult(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    entity_or_relation_id: str
    verification_status: VerificationStatus
    evidence_source: str | None = None
    evidence_snippet: str | None = None
    confidence_delta: float = Field(default=0.0, ge=-1.0, le=1.0)
    verification_method: Literal[
        "direct_lookup",
        "junction_check",
        "text_match",
        "rag_rerank",
        "not_applicable",
    ]


class MindMapNode(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    id: str
    label: str
    entity_type: EntityType
    color: str
    size: int = Field(ge=8, le=60)
    confidence: float = Field(ge=0.0, le=1.0)
    verification_status: VerificationStatus
    source_url: HttpUrl | None = None
    source_table: str
    source_id: str
    tooltip: str
    cluster: str


class MindMapEdge(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    source: str
    target: str
    label: RelationType
    weight: float = Field(ge=0.1, le=5.0)
    color: str
    dashes: bool = False


class MindMapGraph(BaseModel):
    nodes: list[MindMapNode] = Field(default_factory=list)
    edges: list[MindMapEdge] = Field(default_factory=list)
    root_node_id: str | None = None
    query: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    total_sources: int = 0


class AgentStep(BaseModel):
    agent: str
    message: str
    duration_ms: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MindMapState(TypedDict, total=False):
    query: str
    refined_query: str
    collections: list[RetrievalCollection]
    filters: dict[str, Any]
    max_nodes: int
    min_confidence: float
    extraction_mode: str
    retrieved_docs: list[RetrievedDoc]
    extracted_entities: list[Entity]
    extracted_relations: list[Relation]
    verification_results: list[VerificationResult]
    mindmap_graph: MindMapGraph | None
    citations: list[Citation]
    confidence_scores: dict[str, float]
    agent_trace: list[AgentStep]
    messages: Annotated[list[Any], add_messages]
    error: str | None
