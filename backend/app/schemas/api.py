from typing import Any

from pydantic import BaseModel, Field

from backend.app.agents.state import (
    AgentStep,
    Citation,
    Entity,
    MindMapGraph,
    RetrievalCollection,
    RetrievedDoc,
    VerificationResult,
)


class HealthResponse(BaseModel):
    status: str
    app_env: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=2)
    collections: list[RetrievalCollection] = Field(
        default_factory=lambda: [RetrievalCollection.PAPERS, RetrievalCollection.RESOURCES]
    )
    top_k: int = Field(default=10, ge=1, le=100)
    filters: dict[str, Any] = Field(default_factory=dict)
    mode: str = Field(default="hybrid", pattern="^(hybrid|vector|keyword)$")


class SearchResponse(BaseModel):
    results: list[RetrievedDoc]
    total: int


class VerifyRequest(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
    relations: list[dict[str, Any]] = Field(default_factory=list)


class VerifyResponse(BaseModel):
    results: list[VerificationResult]


class MindMapRequest(BaseModel):
    query: str = Field(min_length=2)
    max_nodes: int = Field(default=50, ge=1, le=250)
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    collections: list[RetrievalCollection] = Field(
        default_factory=lambda: [
            RetrievalCollection.PAPERS,
            RetrievalCollection.RESOURCES,
            RetrievalCollection.PROGRAMS,
            RetrievalCollection.COMMUNITIES,
        ]
    )
    filters: dict[str, Any] = Field(default_factory=dict)
    stream: bool = False
    extraction_mode: str = Field(
        default="auto",
        pattern="^(auto|llm|deterministic)$",
    )
    retrieval_mode: str = Field(
        default="hybrid",
        pattern="^(hybrid|vector|keyword)$",
    )


class MindMapResponse(BaseModel):
    graph: MindMapGraph
    citations: list[Citation]
    agent_trace: list[AgentStep]
    total_sources_queried: int
    processing_time_ms: int
    cached: bool = False


class IndexRequest(BaseModel):
    collection: RetrievalCollection
    mode: str = Field(default="incremental", pattern="^(incremental|full)$")
    batch_size: int = Field(default=500, ge=1, le=2_000)
    limit: int | None = Field(default=None, ge=1, le=50_000)


class IndexResponse(BaseModel):
    collection: RetrievalCollection
    status: str
    indexed: int
    skipped: int = 0


class EducatorEnrichRequest(BaseModel):
    query: str
    role: str | None = None
    graph: dict[str, Any]


class CurriculumRequest(BaseModel):
    query: str
    role: str | None = None
    graph: dict[str, Any]


class GapFinderRequest(BaseModel):
    topic_keywords: str = "opioid OR substance OR behavioral OR interprofessional"
    topic: str | None = None
    county: str | None = None
    state: str = "GA"
    limit: int = Field(default=20, ge=1, le=159)


class TeachingPackRequest(BaseModel):
    query: str
    role: str | None = None
    graph: dict[str, Any]
    format: str = Field(default="markdown", pattern="^(json|markdown|docx)$")


class FacultyReviewRequest(BaseModel):
    record_type: str
    record_id: str
    record_title: str | None = None
    decision: str = Field(pattern="^(Useful|Not relevant|Needs review)$")
    reviewer: str = "faculty_user"
    notes: str | None = None
