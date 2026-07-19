from __future__ import annotations

import json
from typing import Any

import structlog

from backend.app.services.educator import ROLE_LABELS, EducatorRole
from backend.app.services.llm_json import call_openai_json

logger = structlog.get_logger(__name__)

MAX_CONTEXT_NODES = 25
MAX_HISTORY_MESSAGES = 6
TOOLTIP_CHARS = 300

_NO_ITEMS_RESPONSE: dict[str, Any] = {
    "answer": "There's nothing on this map yet to answer from -- build a teaching map first.",
    "cited_node_ids": [],
    "grounded": False,
}
_UNAVAILABLE_RESPONSE: dict[str, Any] = {
    "answer": "I couldn't reach the AI assistant right now. Try again in a moment.",
    "cited_node_ids": [],
    "grounded": False,
}


async def answer_question(
    graph: dict[str, Any],
    query: str,
    history: list[dict[str, str]],
    role: str | None,
    api_key: str,
    model: str,
) -> dict[str, Any]:
    """Answer a follow-up question grounded only in the items already shown on
    the current teaching map -- a NotebookLM-style "chat with your sources,"
    scoped to what's already been retrieved and verified rather than a fresh
    open-ended search.
    """
    nodes = graph.get("nodes") or []
    if not nodes:
        return dict(_NO_ITEMS_RESPONSE)

    # Validate before the failure-swallowing try/except below, so a bad request
    # (e.g. an invalid role) surfaces as a normal error instead of being
    # misreported as a transient AI outage.
    role_label = "Educator"
    if role:
        role_label = ROLE_LABELS[EducatorRole(role)]

    ranked_nodes = sorted(nodes, key=lambda node: float(node.get("confidence") or 0.0), reverse=True)
    context_nodes = ranked_nodes[:MAX_CONTEXT_NODES]
    valid_node_ids = {str(node.get("id")) for node in nodes}

    context_payload = [
        {
            "id": node.get("id"),
            "label": node.get("label"),
            "entity_type": node.get("entity_type"),
            "verification_status": node.get("verification_status"),
            "summary": str(node.get("tooltip") or "")[:TOOLTIP_CHARS],
        }
        for node in context_nodes
    ]
    trimmed_history = history[-MAX_HISTORY_MESSAGES:]

    system_prompt = (
        f"You are a research assistant answering questions for a {role_label} about "
        "a teaching map they've already built. Answer ONLY using the items provided "
        "below -- never use outside knowledge, even if you know the general topic. "
        "If the question isn't clearly covered by these items, say so directly "
        "instead of speculating or guessing. "
        'Return JSON: {"answer": str, "cited_node_ids": [str], "grounded": bool}. '
        "cited_node_ids must only contain ids that appear in the provided items."
    )
    user_prompt = json.dumps(
        {
            "items": context_payload,
            "conversation_history": trimmed_history,
            "question": query,
        }
    )

    try:
        data = await call_openai_json(
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
            max_tokens=700,
        )
    except Exception as exc:  # noqa: BLE001 - a chat failure must not break the conversation
        logger.warning("chat_llm_failed", error=str(exc))
        return dict(_UNAVAILABLE_RESPONSE)

    raw_cited = data.get("cited_node_ids") or []
    cited_node_ids = [str(node_id) for node_id in raw_cited if str(node_id) in valid_node_ids]
    grounded = bool(data.get("grounded", False))
    if grounded and not cited_node_ids:
        if raw_cited:
            logger.warning("chat_citations_all_hallucinated", raw_cited_node_ids=raw_cited)
        grounded = False

    answer = str(data.get("answer") or "").strip() or "I wasn't able to generate an answer for that."
    return {"answer": answer, "cited_node_ids": cited_node_ids, "grounded": grounded}
