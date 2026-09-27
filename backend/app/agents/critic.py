import json

import structlog
from langchain_core.runnables import RunnableConfig

from backend.app.agents.state import (
    AgentStep,
    MindMapEdge,
    MindMapState,
    Relation,
    VerificationResult,
    VerificationStatus,
)
from backend.app.services.confidence import apply_verification_multiplier
from backend.app.services.llm_json import call_openai_json

logger = structlog.get_logger(__name__)

MAX_EDGES_TO_CHECK = 8
# Relations worth a semantic second look, regardless of extraction method.
RISKY_VERIFICATION_STATUSES = {
    VerificationStatus.INFERRED,
    VerificationStatus.UNVERIFIED,
    VerificationStatus.CONTESTED,
}
RISKY_VERIFICATION_METHODS = {"not_applicable", "text_match"}


async def critic_node(state: MindMapState, config: RunnableConfig | None = None) -> MindMapState:
    """Semantically double-check high-risk relations using the LLM to compare
    the claim against its evidence text. Can only downgrade an edge to
    CONTESTED, never promote one to CONFIRMED.

    Activation is validation-risk-based, not extraction-mode-based: the critic
    evaluates unreviewed / inferred / free-text / semantically mapped relations
    even when extraction was deterministic.
    """
    configurable = (config or {}).get("configurable", {})
    services = configurable.get("services") if isinstance(configurable, dict) else None

    graph = state.get("mindmap_graph")
    if graph is None:
        return state

    api_key = getattr(getattr(services, "settings", None), "openai_api_key", None)
    secret = api_key.get_secret_value().strip() if api_key is not None else ""

    if not secret:
        return {
            **state,
            "agent_trace": [
                *state.get("agent_trace", []),
                AgentStep(
                    agent="critic",
                    message="Skipped: no OpenAI API key configured for critique.",
                ),
            ],
        }

    # Everything below is best-effort polish on top of an already-complete,
    # successful map. Unlike the earlier pipeline stages, a bug or transient
    # failure here must degrade to "skip the critique," never to state["error"]
    # -- reaching this node at all means research/extraction/verification/mindmap
    # already succeeded, and the guard wrapper in graph.py would otherwise treat
    # any uncaught exception here as a fatal pipeline failure, discarding a good map.
    try:
        verification_by_id = {
            result.entity_or_relation_id: result for result in state.get("verification_results", [])
        }
        relations_by_id = {
            relation.relation_id: relation for relation in state.get("extracted_relations", [])
        }

        candidates: list[tuple[MindMapEdge, Relation, VerificationResult]] = []
        for edge in graph.edges:
            if not edge.relation_id:
                continue
            result = verification_by_id.get(edge.relation_id)
            relation = relations_by_id.get(edge.relation_id)
            if result is None or relation is None:
                continue
            risky_status = result.verification_status in RISKY_VERIFICATION_STATUSES
            risky_method = result.verification_method in RISKY_VERIFICATION_METHODS
            if not (risky_status or risky_method):
                continue
            # Confirmed/database-supported via referential join does not need critique
            if (
                result.verification_status == VerificationStatus.CONFIRMED
                and result.verification_method not in RISKY_VERIFICATION_METHODS
            ):
                continue
            candidates.append((edge, relation, result))

        candidates.sort(key=lambda item: item[0].weight, reverse=True)
        selected = candidates[:MAX_EDGES_TO_CHECK]

        if not selected:
            return {
                **state,
                "agent_trace": [
                    *state.get("agent_trace", []),
                    AgentStep(
                        agent="critic",
                        message="Skipped: no high-risk (unreviewed/inferred/free-text) connections needed critique.",
                    ),
                ],
            }

        nodes_by_id = {node.id: node.label for node in graph.nodes}
        verdicts = await _critique(selected, nodes_by_id, services.settings.openai_model, secret)

        contested_ids: set[str] = set()
        updated_edges: list[MindMapEdge] = []
        for edge in graph.edges:
            verdict = verdicts.get(edge.relation_id) if edge.relation_id else None
            if verdict is None or verdict["verdict"] != "contradicts":
                updated_edges.append(edge)
                continue
            relation = relations_by_id[edge.relation_id]
            new_weight = max(
                0.1,
                min(5.0, apply_verification_multiplier(relation.confidence, VerificationStatus.CONTESTED) * 5.0),
            )
            updated_edges.append(
                edge.model_copy(
                    update={
                        "color": "#8E44AD",
                        "dashes": True,
                        "weight": new_weight,
                        "note": verdict["rationale"],
                        "verification_status": VerificationStatus.CONTESTED,
                    }
                )
            )
            contested_ids.add(edge.relation_id)

        # Propagate Contested into verification_results (not only edge visuals).
        updated_verification = list(state.get("verification_results", []))
        if contested_ids:
            rewritten: list[VerificationResult] = []
            for result in updated_verification:
                if result.entity_or_relation_id in contested_ids:
                    rewritten.append(
                        result.model_copy(
                            update={
                                "verification_status": VerificationStatus.CONTESTED,
                                "evidence_snippet": (
                                    f"Critic contested: "
                                    f"{verdicts.get(result.entity_or_relation_id, {}).get('rationale', '')}"
                                )[:500]
                                or result.evidence_snippet,
                            }
                        )
                    )
                else:
                    rewritten.append(result)
            updated_verification = rewritten

        updated_graph = graph.model_copy(update={"edges": updated_edges})

        return {
            **state,
            "mindmap_graph": updated_graph,
            "verification_results": updated_verification,
            "agent_trace": [
                *state.get("agent_trace", []),
                AgentStep(
                    agent="critic",
                    message=(
                        f"Double-checked {len(selected)} unconfirmed connections; "
                        f"{len(contested_ids)} contested."
                    ),
                    metadata={"checked": len(selected), "contested": len(contested_ids)},
                ),
            ],
        }
    except Exception as exc:  # noqa: BLE001 - a critic failure must not discard an otherwise-good map
        logger.warning("critic_failed", error=str(exc))
        return {
            **state,
            "agent_trace": [
                *state.get("agent_trace", []),
                AgentStep(
                    agent="critic",
                    message=f"Skipped after an error double-checking connections: {exc}",
                ),
            ],
        }


async def _critique(
    selected: list[tuple[MindMapEdge, Relation, VerificationResult]],
    nodes_by_id: dict[str, str],
    model: str,
    api_key: str,
) -> dict[str, dict[str, str]]:
    payload = [
        {
            "index": index,
            "source_label": nodes_by_id.get(edge.source, edge.source),
            "relation_type": str(edge.label),
            "target_label": nodes_by_id.get(edge.target, edge.target),
            "evidence_text": relation.evidence_text[:400],
        }
        for index, (edge, relation, _result) in enumerate(selected)
    ]
    prompt = (
        "You are fact-checking claims extracted for an education knowledge graph. "
        "For each claim, decide whether its evidence_text supports, contradicts, or "
        "is unclear about the claimed relationship between source_label and "
        "target_label. Only use \"contradicts\" when the evidence actively conflicts "
        "with or fails to relate to the claim -- if the evidence is merely thin or "
        "generic, use \"unclear\" instead. "
        'Return JSON: {"verdicts": [{"index": int, "verdict": "supports"|"contradicts"|"unclear", "rationale": str}]}. '
        f"Claims: {json.dumps(payload)}"
    )

    data = await call_openai_json(
        api_key=api_key,
        model=model,
        system_prompt="You are a careful, conservative fact-checker. Reply as strict JSON.",
        user_prompt=prompt,
        temperature=0.0,
        max_tokens=800,
    )

    verdicts_by_id: dict[str, dict[str, str]] = {}
    for item in data.get("verdicts", []):
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        if not isinstance(index, int) or not (0 <= index < len(selected)):
            continue
        verdict = str(item.get("verdict") or "unclear").strip().lower()
        if verdict not in {"supports", "contradicts", "unclear"}:
            verdict = "unclear"
        edge, _relation, _result = selected[index]
        if edge.relation_id:
            verdicts_by_id[edge.relation_id] = {
                "verdict": verdict,
                "rationale": str(item.get("rationale") or "")[:500],
            }
    return verdicts_by_id
