from collections import defaultdict

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

# Keep teaching maps diverse and readable.
TYPE_QUOTAS: dict[EntityType, int] = {
    EntityType.TOPIC: 2,
    EntityType.PAPER: 4,
    EntityType.RESOURCE: 5,
    EntityType.PROGRAM: 4,
    EntityType.COUNTY: 4,
    EntityType.SIMULATION_CASE: 3,
    EntityType.COMPETENCY: 3,
    EntityType.SHORTAGE_AREA: 2,
    EntityType.INSTITUTION: 2,
    EntityType.DISCIPLINE: 2,
    EntityType.AUTHOR: 2,
}

STATUS_RANK = {
    VerificationStatus.CONFIRMED: 0,
    VerificationStatus.INFERRED: 1,
    VerificationStatus.UNVERIFIED: 2,
    VerificationStatus.REFUTED: 3,
}


async def mindmap_node(state: MindMapState) -> MindMapState:
    verification_by_id = {
        result.entity_or_relation_id: result for result in state.get("verification_results", [])
    }
    min_confidence = float(state.get("min_confidence") or 0.0)
    max_nodes = int(state.get("max_nodes") or 50)

    candidates = [
        _node_from_entity(entity, verification_by_id.get(entity.entity_id))
        for entity in state.get("extracted_entities", [])
        if _status(verification_by_id.get(entity.entity_id)) != VerificationStatus.REFUTED
    ]
    nodes = _prune_nodes(candidates, min_confidence=min_confidence, max_nodes=max_nodes)
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
                metadata={
                    "min_confidence": min_confidence,
                    "max_nodes": max_nodes,
                    "pruned_from": len(candidates),
                },
            ),
        ],
    }


def _prune_nodes(
    nodes: list[MindMapNode],
    *,
    min_confidence: float,
    max_nodes: int,
) -> list[MindMapNode]:
    if not nodes:
        return []

    topics = [node for node in nodes if node.entity_type == EntityType.TOPIC]
    others = [node for node in nodes if node.entity_type != EntityType.TOPIC]

    def sort_key(node: MindMapNode) -> tuple[int, int, float]:
        return (
            STATUS_RANK.get(VerificationStatus(node.verification_status), 9),
            0 if node.confidence >= min_confidence else 1,
            -float(node.confidence),
        )

    others_sorted = sorted(others, key=sort_key)
    selected: list[MindMapNode] = []
    # Always keep topic roots first.
    selected.extend(sorted(topics, key=sort_key)[: TYPE_QUOTAS[EntityType.TOPIC]])

    per_type: dict[EntityType, int] = defaultdict(int)
    for node in selected:
        per_type[EntityType(node.entity_type)] += 1

    for node in others_sorted:
        entity_type = EntityType(node.entity_type)
        # Soft floor: keep CONFIRMED items near threshold; drop weak UNVERIFIED.
        if node.confidence < min_confidence:
            if VerificationStatus(node.verification_status) != VerificationStatus.CONFIRMED:
                continue
            if node.confidence < max(0.25, min_confidence - 0.15):
                continue
        quota = TYPE_QUOTAS.get(entity_type, 3)
        if per_type[entity_type] >= quota:
            continue
        selected.append(node)
        per_type[entity_type] += 1
        if len(selected) >= max_nodes:
            break

    # If still under max_nodes, fill with best remaining confirmed/inferred items.
    if len(selected) < max_nodes:
        selected_ids = {node.id for node in selected}
        for node in others_sorted:
            if node.id in selected_ids:
                continue
            if node.confidence < min_confidence and VerificationStatus(
                node.verification_status
            ) != VerificationStatus.CONFIRMED:
                continue
            selected.append(node)
            if len(selected) >= max_nodes:
                break

    return selected


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
