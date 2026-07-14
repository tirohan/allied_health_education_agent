from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig

from backend.app.agents.state import (
    AgentStep,
    Entity,
    EntityType,
    MindMapState,
    Relation,
    RelationType,
    VerificationResult,
    VerificationStatus,
)
from backend.app.db.postgres import Postgres
from backend.app.services.confidence import confidence_delta

VerificationMethod = Literal[
    "direct_lookup",
    "junction_check",
    "text_match",
    "rag_rerank",
    "not_applicable",
]
EntityVerifier = Callable[[Postgres, Entity], Awaitable[VerificationResult]]
RelationVerifier = Callable[[Postgres, Relation, Mapping[str, Entity]], Awaitable[VerificationResult]]


ENTITY_LOOKUPS: dict[EntityType, tuple[str, str, str]] = {
    EntityType.TOPIC: ("topic_modules", "topic_tag", "topic_label"),
    EntityType.COMPETENCY: ("competencies", "competency_id", "competency_name"),
    EntityType.DISCIPLINE: ("disciplines", "cip_code", "cip_title"),
    EntityType.PAPER: ("research_papers", "paper_id", "title"),
    EntityType.RESOURCE: ("resources", "resource_id", "title"),
    EntityType.PROGRAM: ("programs", "program_id", "program_title"),
    EntityType.INSTITUTION: ("institutions", "unitid", "institution_name"),
    EntityType.COUNTY: ("county_profiles", "county_fips", "county_name"),
    EntityType.SHORTAGE_AREA: ("workforce_shortage_records", "record_id", "designation_name"),
    EntityType.SIMULATION_CASE: ("simulation_cases", "simulation_case_id", "title"),
    EntityType.AUTHOR: ("authors", "author_id", "display_name"),
}


async def verify_entity(db: Postgres, entity: Entity) -> VerificationResult:
    lookup = ENTITY_LOOKUPS.get(EntityType(entity.entity_type))
    if lookup is None:
        return _result(
            entity.entity_id,
            VerificationStatus.UNVERIFIED,
            entity.confidence,
            "not_applicable",
            None,
            "Unsupported entity type",
        )

    table, key_column, label_column = lookup
    if entity.source_table != table:
        return _result(
            entity.entity_id,
            VerificationStatus.REFUTED,
            entity.confidence,
            "direct_lookup",
            f"{table}.{key_column}",
            f"Expected source_table={table}; got {entity.source_table}",
        )

    row = await db.fetchrow(
        f"SELECT {key_column}, {label_column} FROM {table} WHERE {key_column} = $1 LIMIT 1",
        _coerce_key(entity.source_id, EntityType(entity.entity_type)),
    )
    if row is None:
        return _result(
            entity.entity_id,
            VerificationStatus.REFUTED,
            entity.confidence,
            "direct_lookup",
            f"{table}.{key_column}",
            "No matching source record was found",
        )

    label = str(row.get(label_column) or "")
    status = VerificationStatus.CONFIRMED
    if label and entity.label:
        entity_label = entity.label.lower()
        record_label = label.lower()
        if (
            entity_label not in record_label
            and record_label not in entity_label
            and not any(token and token in record_label for token in entity_label.split()[:2])
        ):
            status = VerificationStatus.INFERRED

    return _result(
        entity.entity_id,
        status,
        entity.confidence,
        "direct_lookup",
        f"{table}.{key_column}",
        label[:500] or "Record exists",
    )


async def verify_relation(
    db: Postgres,
    relation: Relation,
    entities_by_id: Mapping[str, Entity],
) -> VerificationResult:
    source = entities_by_id.get(relation.source_entity_id)
    target = entities_by_id.get(relation.target_entity_id)
    if source is None or target is None:
        return _result(
            relation.relation_id,
            VerificationStatus.UNVERIFIED,
            relation.confidence,
            "junction_check",
            None,
            "Source or target entity is missing from extraction output",
        )

    relation_type = RelationType(relation.relation_type)
    verifier = RELATION_VERIFIERS.get(relation_type)
    if verifier is None:
        return _result(
            relation.relation_id,
            VerificationStatus.INFERRED,
            relation.confidence,
            "not_applicable",
            None,
            "No direct junction rule exists for this relation type",
        )
    return await verifier(db, relation, entities_by_id)


async def verify_all(
    db: Postgres,
    entities: list[Entity],
    relations: list[Relation],
) -> list[VerificationResult]:
    entity_results = [await verify_entity(db, entity) for entity in entities]
    entities_by_id = {entity.entity_id: entity for entity in entities}
    relation_results = [
        await verify_relation(db, relation, entities_by_id) for relation in relations
    ]
    return entity_results + relation_results


async def verification_node(state: MindMapState, config: RunnableConfig) -> MindMapState:
    services = config["configurable"]["services"]
    results = await verify_all(
        services.postgres,
        state.get("extracted_entities", []),
        state.get("extracted_relations", []),
    )
    status_counts: dict[str, int] = {}
    for result in results:
        status = str(result.verification_status)
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        **state,
        "verification_results": results,
        "agent_trace": [
            *state.get("agent_trace", []),
            AgentStep(
                agent="verification",
                message=f"Verified {len(results)} entities and relations.",
                metadata=status_counts,
            ),
        ],
    }


async def _verify_mapped_to(
    db: Postgres,
    relation: Relation,
    entities_by_id: Mapping[str, Entity],
) -> VerificationResult:
    source = entities_by_id[relation.source_entity_id]
    target = entities_by_id[relation.target_entity_id]
    if EntityType(source.entity_type) == EntityType.RESOURCE and EntityType(target.entity_type) == EntityType.TOPIC:
        row = await db.fetchrow(
            """
            SELECT resource_id, topic_tag, topic_source
            FROM resource_topics
            WHERE resource_id = $1 AND topic_tag = $2
            LIMIT 1
            """,
            source.source_id,
            target.source_id,
        )
        return _junction_result(relation, row, "resource_topics.resource_id/topic_tag")

    return _result(
        relation.relation_id,
        VerificationStatus.INFERRED,
        relation.confidence,
        "not_applicable",
        None,
        "MAPPED_TO relation has no direct rule for this entity pair",
    )


async def _verify_addresses(
    db: Postgres,
    relation: Relation,
    entities_by_id: Mapping[str, Entity],
) -> VerificationResult:
    source = entities_by_id[relation.source_entity_id]
    target = entities_by_id[relation.target_entity_id]
    if EntityType(source.entity_type) == EntityType.RESOURCE and EntityType(target.entity_type) == EntityType.COMPETENCY:
        row = await db.fetchrow(
            """
            SELECT resource_id, competency_id, alignment_strength
            FROM resource_competencies
            WHERE resource_id = $1 AND competency_id = $2
            LIMIT 1
            """,
            source.source_id,
            target.source_id,
        )
        return _junction_result(relation, row, "resource_competencies.resource_id/competency_id")

    if EntityType(source.entity_type) == EntityType.PAPER and EntityType(target.entity_type) == EntityType.COMPETENCY:
        row = await db.fetchrow(
            """
            SELECT paper_id, competency_id, match_score
            FROM paper_competencies
            WHERE paper_id = $1 AND competency_id = $2
            LIMIT 1
            """,
            source.source_id,
            target.source_id,
        )
        return _junction_result(relation, row, "paper_competencies.paper_id/competency_id")

    return _result(
        relation.relation_id,
        VerificationStatus.INFERRED,
        relation.confidence,
        "not_applicable",
        None,
        "ADDRESSES relation has no direct rule for this entity pair",
    )


async def _verify_authored_by(
    db: Postgres,
    relation: Relation,
    entities_by_id: Mapping[str, Entity],
) -> VerificationResult:
    source = entities_by_id[relation.source_entity_id]
    target = entities_by_id[relation.target_entity_id]
    row = await db.fetchrow(
        """
        SELECT paper_id, author_id, author_position
        FROM paper_authors
        WHERE paper_id = $1 AND author_id = $2
        LIMIT 1
        """,
        source.source_id,
        target.source_id,
    )
    return _junction_result(relation, row, "paper_authors.paper_id/author_id")


async def _verify_offered_at(
    db: Postgres,
    relation: Relation,
    entities_by_id: Mapping[str, Entity],
) -> VerificationResult:
    source = entities_by_id[relation.source_entity_id]
    target = entities_by_id[relation.target_entity_id]
    row = await db.fetchrow(
        """
        SELECT program_id, unitid
        FROM programs
        WHERE program_id = $1 AND unitid = $2
        LIMIT 1
        """,
        source.source_id,
        int(target.source_id),
    )
    return _junction_result(relation, row, "programs.program_id/unitid")


RELATION_VERIFIERS: dict[RelationType, RelationVerifier] = {
    RelationType.MAPPED_TO: _verify_mapped_to,
    RelationType.ADDRESSES: _verify_addresses,
    RelationType.AUTHORED_BY: _verify_authored_by,
    RelationType.OFFERED_AT: _verify_offered_at,
}


def _junction_result(
    relation: Relation,
    row: Mapping[str, Any] | None,
    evidence_source: str,
) -> VerificationResult:
    if row is None:
        return _result(
            relation.relation_id,
            VerificationStatus.UNVERIFIED,
            relation.confidence,
            "junction_check",
            evidence_source,
            "No direct junction record was found",
        )
    return _result(
        relation.relation_id,
        VerificationStatus.CONFIRMED,
        relation.confidence,
        "junction_check",
        evidence_source,
        "; ".join(f"{key}={value}" for key, value in row.items())[:500],
    )


def _result(
    item_id: str,
    status: VerificationStatus,
    confidence: float,
    method: VerificationMethod,
    evidence_source: str | None,
    evidence_snippet: str | None,
) -> VerificationResult:
    return VerificationResult(
        entity_or_relation_id=item_id,
        verification_status=status,
        evidence_source=evidence_source,
        evidence_snippet=evidence_snippet,
        confidence_delta=confidence_delta(confidence, status),
        verification_method=method,
    )


def _coerce_key(source_id: str, entity_type: EntityType) -> object:
    if entity_type == EntityType.INSTITUTION:
        return int(source_id)
    return source_id
