import json
from unittest.mock import AsyncMock, patch

import pytest

from backend.app.services.chat import MAX_CONTEXT_NODES, answer_question


def _node(node_id: str, confidence: float) -> dict:
    return {
        "id": node_id,
        "label": f"Item {node_id}",
        "entity_type": "Paper",
        "verification_status": "CONFIRMED",
        "confidence": confidence,
        "tooltip": "Some evidence text.",
    }


@pytest.mark.asyncio
async def test_answer_question_empty_graph_skips_llm_call() -> None:
    with patch("backend.app.services.chat.call_openai_json", new_callable=AsyncMock) as mock_call:
        result = await answer_question({"nodes": []}, "any question", [], None, "sk-test", "gpt-4o")
    mock_call.assert_not_called()
    assert result["grounded"] is False
    assert result["cited_node_ids"] == []


@pytest.mark.asyncio
async def test_answer_question_invalid_role_raises_before_llm_call() -> None:
    graph = {"nodes": [_node("paper:1", 0.9)]}
    with patch("backend.app.services.chat.call_openai_json", new_callable=AsyncMock) as mock_call:
        with pytest.raises(ValueError):
            await answer_question(graph, "q", [], "not_a_real_role", "sk-test", "gpt-4o")
    mock_call.assert_not_called()


@pytest.mark.asyncio
async def test_answer_question_caps_context_to_top_confidence_nodes() -> None:
    nodes = [_node(f"paper:{i}", confidence=i / 100) for i in range(1, 41)]  # 40 nodes
    graph = {"nodes": nodes}
    canned = {"answer": "ok", "cited_node_ids": [], "grounded": False}
    with patch(
        "backend.app.services.chat.call_openai_json", new_callable=AsyncMock, return_value=canned
    ) as mock_call:
        await answer_question(graph, "q", [], None, "sk-test", "gpt-4o")

    sent_user_prompt = mock_call.call_args.kwargs["user_prompt"]
    sent_items = json.loads(sent_user_prompt)["items"]
    assert len(sent_items) == MAX_CONTEXT_NODES
    # Highest-confidence nodes (paper:40 down to paper:16) should be the ones sent.
    sent_ids = {item["id"] for item in sent_items}
    assert "paper:40" in sent_ids
    assert "paper:1" not in sent_ids


@pytest.mark.asyncio
async def test_answer_question_filters_hallucinated_citations_and_downgrades_grounded() -> None:
    graph = {"nodes": [_node("paper:1", 0.9)]}
    canned = {
        "answer": "This is supported.",
        "cited_node_ids": ["paper:999"],  # not a real node id
        "grounded": True,
    }
    with patch(
        "backend.app.services.chat.call_openai_json", new_callable=AsyncMock, return_value=canned
    ):
        result = await answer_question(graph, "q", [], None, "sk-test", "gpt-4o")

    assert result["cited_node_ids"] == []
    assert result["grounded"] is False


@pytest.mark.asyncio
async def test_answer_question_keeps_valid_citations() -> None:
    graph = {"nodes": [_node("paper:1", 0.9), _node("paper:2", 0.5)]}
    canned = {
        "answer": "Paper 1 covers this.",
        "cited_node_ids": ["paper:1", "paper:999"],
        "grounded": True,
    }
    with patch(
        "backend.app.services.chat.call_openai_json", new_callable=AsyncMock, return_value=canned
    ):
        result = await answer_question(graph, "q", [], None, "sk-test", "gpt-4o")

    assert result["cited_node_ids"] == ["paper:1"]
    assert result["grounded"] is True


@pytest.mark.asyncio
async def test_answer_question_gracefully_degrades_on_llm_failure() -> None:
    graph = {"nodes": [_node("paper:1", 0.9)]}
    with patch(
        "backend.app.services.chat.call_openai_json",
        new_callable=AsyncMock,
        side_effect=RuntimeError("boom"),
    ):
        result = await answer_question(graph, "q", [], None, "sk-test", "gpt-4o")

    assert result["grounded"] is False
    assert result["cited_node_ids"] == []
    assert "couldn't reach" in result["answer"].lower()
