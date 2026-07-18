from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from backend.app.agents.critic import critic_node
from backend.app.agents.state import (
    AgentStep,
    MindMapEdge,
    MindMapGraph,
    MindMapNode,
    Relation,
    RelationType,
    VerificationResult,
    VerificationStatus,
)


class _FakeSecret:
    def __init__(self, value: str) -> None:
        self._value = value

    def get_secret_value(self) -> str:
        return self._value


def _services(api_key: str | None = "sk-test") -> SimpleNamespace:
    settings = SimpleNamespace(
        openai_api_key=_FakeSecret(api_key) if api_key else None,
        openai_model="gpt-4o",
    )
    return SimpleNamespace(settings=settings)


def _node(node_id: str, label: str) -> MindMapNode:
    return MindMapNode(
        id=node_id,
        label=label,
        entity_type="Paper",
        color="#50C878",
        size=20,
        confidence=0.7,
        verification_status=VerificationStatus.INFERRED,
        source_table="research_papers",
        source_id=node_id,
        tooltip=label,
        cluster="Paper",
    )


def _edge(relation_id: str, weight: float = 3.0) -> MindMapEdge:
    return MindMapEdge(
        source="paper:1",
        target="query_root",
        label=RelationType.SUPPORTS,
        weight=weight,
        color="#F5A623",
        dashes=True,
        relation_id=relation_id,
    )


def _state(
    *,
    extraction_mode: str,
    edges: list[MindMapEdge],
    verification_results: list[VerificationResult],
    relations: list[Relation],
) -> dict:
    graph = MindMapGraph(
        nodes=[_node("paper:1", "Some Paper"), _node("query_root", "The Topic")],
        edges=edges,
        query="q",
    )
    return {
        "mindmap_graph": graph,
        "verification_results": verification_results,
        "extracted_relations": relations,
        "agent_trace": [
            AgentStep(agent="extraction", message="...", metadata={"mode": extraction_mode})
        ],
    }


def _config(services: SimpleNamespace) -> dict:
    return {"configurable": {"services": services}}


@pytest.mark.asyncio
async def test_critic_skips_when_extraction_not_llm() -> None:
    state = _state(
        extraction_mode="deterministic",
        edges=[_edge("r1")],
        verification_results=[
            VerificationResult(
                entity_or_relation_id="r1",
                verification_status=VerificationStatus.INFERRED,
                verification_method="text_match",
            )
        ],
        relations=[
            Relation(
                relation_id="r1", source_entity_id="paper:1", target_entity_id="query_root",
                relation_type=RelationType.SUPPORTS, confidence=0.7,
            )
        ],
    )
    with patch("backend.app.agents.critic._critique", new_callable=AsyncMock) as mock_critique:
        result = await critic_node(state, _config(_services()))
    mock_critique.assert_not_called()
    assert result["mindmap_graph"].edges[0].note is None
    assert "Skipped" in result["agent_trace"][-1].message


@pytest.mark.asyncio
async def test_critic_skips_when_no_openai_key() -> None:
    state = _state(
        extraction_mode="llm_structured",
        edges=[_edge("r1")],
        verification_results=[
            VerificationResult(
                entity_or_relation_id="r1",
                verification_status=VerificationStatus.INFERRED,
                verification_method="text_match",
            )
        ],
        relations=[
            Relation(
                relation_id="r1", source_entity_id="paper:1", target_entity_id="query_root",
                relation_type=RelationType.SUPPORTS, confidence=0.7,
            )
        ],
    )
    with patch("backend.app.agents.critic._critique", new_callable=AsyncMock) as mock_critique:
        result = await critic_node(state, _config(_services(api_key=None)))
    mock_critique.assert_not_called()
    assert "Skipped" in result["agent_trace"][-1].message


@pytest.mark.asyncio
async def test_critic_skips_when_no_eligible_edges() -> None:
    # verification_method="junction_check" means it was already referentially
    # confirmed -- nothing for the critic to double-check.
    state = _state(
        extraction_mode="llm_structured",
        edges=[_edge("r1")],
        verification_results=[
            VerificationResult(
                entity_or_relation_id="r1",
                verification_status=VerificationStatus.CONFIRMED,
                verification_method="junction_check",
            )
        ],
        relations=[
            Relation(
                relation_id="r1", source_entity_id="paper:1", target_entity_id="query_root",
                relation_type=RelationType.SUPPORTS, confidence=0.7,
            )
        ],
    )
    with patch("backend.app.agents.critic._critique", new_callable=AsyncMock) as mock_critique:
        result = await critic_node(state, _config(_services()))
    mock_critique.assert_not_called()
    assert "Skipped" in result["agent_trace"][-1].message


@pytest.mark.asyncio
async def test_critic_downgrades_only_contradicted_edges() -> None:
    state = _state(
        extraction_mode="llm_structured",
        edges=[_edge("r1"), _edge("r2")],
        verification_results=[
            VerificationResult(
                entity_or_relation_id="r1",
                verification_status=VerificationStatus.INFERRED,
                verification_method="text_match",
            ),
            VerificationResult(
                entity_or_relation_id="r2",
                verification_status=VerificationStatus.INFERRED,
                verification_method="text_match",
            ),
        ],
        relations=[
            Relation(
                relation_id="r1", source_entity_id="paper:1", target_entity_id="query_root",
                relation_type=RelationType.SUPPORTS, confidence=0.7,
            ),
            Relation(
                relation_id="r2", source_entity_id="paper:1", target_entity_id="query_root",
                relation_type=RelationType.SUPPORTS, confidence=0.7,
            ),
        ],
    )
    canned_verdicts = {
        "r1": {"verdict": "contradicts", "rationale": "Evidence does not mention this."},
        "r2": {"verdict": "supports", "rationale": "Evidence backs this up."},
    }
    with patch(
        "backend.app.agents.critic._critique", new_callable=AsyncMock, return_value=canned_verdicts
    ):
        result = await critic_node(state, _config(_services()))

    edges_by_id = {edge.relation_id: edge for edge in result["mindmap_graph"].edges}
    contested = edges_by_id["r1"]
    assert contested.color == "#8E44AD"
    assert contested.dashes is True
    assert contested.note == "Evidence does not mention this."

    unchanged = edges_by_id["r2"]
    assert unchanged.color == "#F5A623"  # never promoted/altered on a "supports" verdict
    assert unchanged.note is None

    trace_message = result["agent_trace"][-1].message
    assert "1 contested" in trace_message


@pytest.mark.asyncio
async def test_critic_gracefully_degrades_on_failure() -> None:
    state = _state(
        extraction_mode="llm_structured",
        edges=[_edge("r1")],
        verification_results=[
            VerificationResult(
                entity_or_relation_id="r1",
                verification_status=VerificationStatus.INFERRED,
                verification_method="text_match",
            )
        ],
        relations=[
            Relation(
                relation_id="r1", source_entity_id="paper:1", target_entity_id="query_root",
                relation_type=RelationType.SUPPORTS, confidence=0.7,
            )
        ],
    )
    with patch(
        "backend.app.agents.critic._critique", new_callable=AsyncMock, side_effect=RuntimeError("boom")
    ):
        result = await critic_node(state, _config(_services()))

    # No exception propagated, and the original (unmodified) graph survives.
    assert result["mindmap_graph"].edges[0].note is None
    assert "error" not in result
    assert "Skipped" in result["agent_trace"][-1].message
