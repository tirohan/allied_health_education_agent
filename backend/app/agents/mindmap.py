from backend.app.agents.state import (
    AgentStep,
    Entity,
    EntityType,
    MindMapEdge,
    MindMapGraph,
    MindMapNode,
    MindMapState,
    Relation,
    VerificationResult,
    VerificationStatus,
)
from backend.app.services.confidence import apply_verification_multiplier


ENTITY_COLORS: dict[EntityType, str] = {
    EntityType.TOPIC: "#4A90D9",
    EntityType.PAPER: "#50C878",
    EntityType.RESOURCE: "#F5A623",
    EntityType.PROGRAM: "#9B59B6",
    EntityType.COMPETENCY: "#E74C3C",
    EntityType.COUNTY: "#1ABC9C",
    EntityType.SHORTAGE_AREA: "#E67E22",
    EntityType.SIMULATION_CASE: "#2ECC71",
    EntityType.INSTITUTION: "#95A5A6",
    EntityType.DISCIPLINE: "#34495E",
    EntityType.AUTHOR: "#7F8C8D",
}

STATUS_COLORS: dict[VerificationStatus, str] = {
    VerificationStatus.CONFIRMED: "#50C878",
    VerificationStatus.INFERRED: "#F5A623",
    VerificationStatus.UNVERIFIED: "#95A5A6",
    VerificationStatus.REFUTED: "#E74C3C",
}


async def mindmap_node(state: MindMapState) -> MindMapState:
    verification_by_id = {
        result.entity_or_relation_id: result for result in state.get("verification_results", [])
    }
    nodes = [
        _node_from_entity(entity, verification_by_id.get(entity.entity_id))
        for entity in state.get("extracted_entities", [])
        if _status(verification_by_id.get(entity.entity_id)) != VerificationStatus.REFUTED
    ]
    node_ids = {node.id for node in nodes}
    edges = [
        _edge_from_relation(relation, verification_by_id.get(relation.relation_id))
        for relation in state.get("extracted_relations", [])
        if relation.source_entity_id in node_ids
        and relation.target_entity_id in node_ids
        and _status(verification_by_id.get(relation.relation_id)) != VerificationStatus.REFUTED
    ]
    graph = MindMapGraph(
        nodes=nodes,
        edges=edges,
        root_node_id="query_root" if "query_root" in node_ids else (nodes[0].id if nodes else None),
        query=state["query"],
        total_sources=len(state.get("retrieved_docs", [])),
    )
    return {
        **state,
        "mindmap_graph": graph,
        "agent_trace": [
            *state.get("agent_trace", []),
            AgentStep(
                agent="mindmap",
                message=f"Built graph with {len(nodes)} nodes and {len(edges)} edges.",
            ),
        ],
    }


def _node_from_entity(entity: Entity, verification: VerificationResult | None) -> MindMapNode:
    status = _status(verification)
    confidence = apply_verification_multiplier(entity.confidence, status)
    tooltip = entity.summary or entity.label
    if verification and verification.evidence_snippet:
        tooltip = f"{tooltip}\n\nEvidence: {verification.evidence_snippet}"
    return MindMapNode(
        id=entity.entity_id,
        label=entity.label,
        entity_type=EntityType(entity.entity_type),
        color=ENTITY_COLORS[EntityType(entity.entity_type)],
        size=max(8, min(60, int(confidence * 40) + 8)),
        confidence=confidence,
        verification_status=status,
        source_table=entity.source_table,
        source_id=entity.source_id,
        tooltip=tooltip[:1_000],
        cluster=EntityType(entity.entity_type).value,
    )


def _edge_from_relation(
    relation: Relation,
    verification: VerificationResult | None,
) -> MindMapEdge:
    status = _status(verification)
    confidence = apply_verification_multiplier(relation.confidence, status)
    return MindMapEdge(
        source=relation.source_entity_id,
        target=relation.target_entity_id,
        label=relation.relation_type,
        weight=max(0.1, min(5.0, confidence * 5.0)),
        color=STATUS_COLORS[status],
        dashes=status in {VerificationStatus.INFERRED, VerificationStatus.UNVERIFIED},
    )


def _status(verification: VerificationResult | None) -> VerificationStatus:
    if verification is None:
        return VerificationStatus.UNVERIFIED
    return VerificationStatus(verification.verification_status)
