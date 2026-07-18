import asyncio
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
    "faculty_review",
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


def _labels_overlap(a: str, b: str) -> bool:
    """Loose overlap check: substring match either way, or shares a leading token.

    Used only where both labels already describe the *same* database row (e.g.
    an entity's own label vs. its own record, matched by primary key) -- a
    shared generic word is a low-risk false positive there. Do not reuse this
    for comparing a topic label against an unrelated free-text classification;
    use `_topic_overlap` instead.
    """
    a_norm = a.strip().lower()
    b_norm = b.strip().lower()
    if not a_norm or not b_norm:
        return False
    if a_norm in b_norm or b_norm in a_norm:
        return True
    return any(token and token in b_norm for token in a_norm.split()[:2])


_GENERIC_TOPIC_WORDS = {
    "and", "or", "the", "of", "for", "in", "on", "to", "a", "an",
    "education", "health", "care", "medical", "training", "learning",
    "healthcare", "practice", "services", "professional",
}


def _significant_words(text: str) -> set[str]:
    words = {word for word in text.replace("-", " ").split() if len(word) > 2}
    return (words - _GENERIC_TOPIC_WORDS) or words


def _topic_overlap(topic_label: str, candidate: str) -> bool:
    """Compare a topic's canonical label against an unrelated free-text
    classification (paper_topics.topic_name, topic_tags_inferred,
    simulation_cases.topic_tags). Two labels that only share a generic word
    like "education" or "health" (verified live against real data to produce
    false matches, e.g. "Interprofessional Education" vs "Medical education")
    must not count as a match -- every significant word of the shorter side
    has to actually appear in the longer text.
    """
    a_norm = topic_label.strip().lower()
    b_norm = candidate.strip().lower()
    if not a_norm or not b_norm:
        return False
    if a_norm == b_norm:
        return True
    a_words = _significant_words(a_norm)
    b_words = _significant_words(b_norm)
    if not a_words or not b_words:
        return False
    shorter, longer_text = (a_words, b_norm) if len(a_words) <= len(b_words) else (b_words, a_norm)
    return all(word in longer_text for word in shorter)


async def _resolve_topic_label(db: Postgres, topic_tag: str) -> str | None:
    row = await db.fetchrow(
        "SELECT topic_label FROM topic_modules WHERE topic_tag = $1",
        topic_tag,
    )
    return str(row["topic_label"]) if row and row.get("topic_label") else None


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

    key = _coerce_key(entity.source_id, EntityType(entity.entity_type))
    if key is None:
        return _result(
            entity.entity_id,
            VerificationStatus.REFUTED,
            entity.confidence,
            "direct_lookup",
            f"{table}.{key_column}",
            f"source_id={entity.source_id!r} is not a valid key for {table}.{key_column}",
        )

    row = await db.fetchrow(
        f"SELECT {key_column}, {label_column} FROM {table} WHERE {key_column} = $1 LIMIT 1",
        key,
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
    if label and entity.label and not _labels_overlap(entity.label, label):
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
    # verify_entity now handles bad source_ids by returning a REFUTED result
    # instead of raising, so it is safe to fan these out concurrently rather
    # than doing one DB round-trip at a time.
    entity_results = await asyncio.gather(
        *[verify_entity(db, entity) for entity in entities],
        return_exceptions=False,
    )
    entities_by_id = {entity.entity_id: entity for entity in entities}
    relation_results = await asyncio.gather(
        *[verify_relation(db, relation, entities_by_id) for relation in relations],
        return_exceptions=False,
    )
    return list(entity_results) + list(relation_results)


async def _apply_faculty_overrides(
    db: Postgres,
    results: list[VerificationResult],
    entities: list[Entity],
) -> list[VerificationResult]:
    """Let stored faculty review decisions override automated verification.

    Without this, `submit_faculty_review` writes are permanently inert -- a
    faculty "Not relevant" vote never affects any future map, even though
    `mindmap_node` already filters out REFUTED entities for free once one
    exists here.
    """
    rows = await db.fetch(
        """
        SELECT DISTINCT ON (record_type, record_id)
            record_type, record_id, verification_status, verified_by, verified_date, notes
        FROM verification_logs
        WHERE evidence_level = 'faculty_review'
        ORDER BY record_type, record_id, verified_date DESC, verification_id DESC
        """
    )
    if not rows:
        return results

    overrides = {(row["record_type"], row["record_id"]): row for row in rows}
    entities_by_id = {entity.entity_id: entity for entity in entities}

    updated: list[VerificationResult] = []
    for result in results:
        entity = entities_by_id.get(result.entity_or_relation_id)
        override = overrides.get((entity.source_table, entity.source_id)) if entity else None
        if entity is None or override is None:
            updated.append(result)
            continue

        # verification_logs.verification_status already stores the mapped
        # enum value (submit_faculty_review's status_map writes "CONFIRMED"/
        # "REFUTED"/"UNVERIFIED", not the raw "Useful"/"Not relevant" decision
        # text), so parse it directly rather than re-mapping from decision text.
        try:
            status = VerificationStatus(str(override["verification_status"]))
        except ValueError:
            updated.append(result)
            continue

        updated.append(
            result.model_copy(
                update={
                    "verification_status": status,
                    "evidence_source": "faculty_review",
                    "evidence_snippet": (
                        f"Faculty ({override['verified_by']}) marked this "
                        f"{override['verified_date']}: {override['notes'] or ''}"
                    )[:500],
                    "confidence_delta": confidence_delta(entity.confidence, status),
                    "verification_method": "faculty_review",
                }
            )
        )
    return updated


async def verification_node(state: MindMapState, config: RunnableConfig) -> MindMapState:
    services = config["configurable"]["services"]
    entities = state.get("extracted_entities", [])
    results = await verify_all(
        services.postgres,
        entities,
        state.get("extracted_relations", []),
    )
    results = await _apply_faculty_overrides(services.postgres, results, entities)
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


async def _verify_supports(
    db: Postgres,
    relation: Relation,
    entities_by_id: Mapping[str, Entity],
) -> VerificationResult:
    source = entities_by_id[relation.source_entity_id]
    target = entities_by_id[relation.target_entity_id]
    if not (
        EntityType(source.entity_type) == EntityType.PAPER
        and EntityType(target.entity_type) == EntityType.TOPIC
    ):
        return _result(
            relation.relation_id,
            VerificationStatus.INFERRED,
            relation.confidence,
            "not_applicable",
            None,
            "SUPPORTS relation has no direct rule for this entity pair",
        )

    topic_label = await _resolve_topic_label(db, target.source_id) or target.label
    topic_rows = await db.fetch(
        "SELECT topic_name FROM paper_topics WHERE paper_id = $1",
        source.source_id,
    )
    if any(_topic_overlap(topic_label, str(row.get("topic_name") or "")) for row in topic_rows):
        return _result(
            relation.relation_id,
            VerificationStatus.CONFIRMED,
            relation.confidence,
            "text_match",
            "paper_topics.paper_id/topic_name",
            f"Paper is topically classified as matching '{topic_label}'",
        )

    inferred_row = await db.fetchrow(
        "SELECT topic_tags_inferred FROM research_papers WHERE paper_id = $1",
        source.source_id,
    )
    inferred_tags = str((inferred_row or {}).get("topic_tags_inferred") or "")
    if inferred_tags and _topic_overlap(topic_label, inferred_tags):
        return _result(
            relation.relation_id,
            VerificationStatus.CONFIRMED,
            relation.confidence,
            "text_match",
            "research_papers.topic_tags_inferred",
            f"Paper's inferred topic tags mention '{inferred_tags}'",
        )

    if topic_rows:
        return _result(
            relation.relation_id,
            VerificationStatus.INFERRED,
            relation.confidence,
            "text_match",
            "paper_topics.paper_id",
            "Paper has topic classifications on file, but none matched this topic directly",
        )
    return _result(
        relation.relation_id,
        VerificationStatus.UNVERIFIED,
        relation.confidence,
        "text_match",
        None,
        "No topic classification records were found for this paper",
    )


async def _verify_trained_for(
    db: Postgres,
    relation: Relation,
    entities_by_id: Mapping[str, Entity],
) -> VerificationResult:
    source = entities_by_id[relation.source_entity_id]
    target = entities_by_id[relation.target_entity_id]
    if not (
        EntityType(source.entity_type) == EntityType.SIMULATION_CASE
        and EntityType(target.entity_type) == EntityType.TOPIC
    ):
        return _result(
            relation.relation_id,
            VerificationStatus.INFERRED,
            relation.confidence,
            "not_applicable",
            None,
            "TRAINED_FOR relation has no direct rule for this entity pair",
        )

    topic_label = await _resolve_topic_label(db, target.source_id) or target.label
    row = await db.fetchrow(
        """
        SELECT simulation_case_id, topic_tag
        FROM case_topics
        WHERE simulation_case_id = $1 AND topic_tag = $2
        LIMIT 1
        """,
        source.source_id,
        target.source_id,
    )
    if row is not None:
        return _junction_result(relation, row, "case_topics.simulation_case_id/topic_tag")

    case_row = await db.fetchrow(
        "SELECT topic_tags FROM simulation_cases WHERE simulation_case_id = $1",
        source.source_id,
    )
    topic_tags = str((case_row or {}).get("topic_tags") or "")
    if topic_tags and _topic_overlap(topic_label, topic_tags):
        return _result(
            relation.relation_id,
            VerificationStatus.INFERRED,
            relation.confidence,
            "text_match",
            "simulation_cases.topic_tags",
            f"Case's topic tags mention '{topic_tags}'",
        )
    return _result(
        relation.relation_id,
        VerificationStatus.UNVERIFIED,
        relation.confidence,
        "text_match",
        None,
        "No topic classification records were found for this simulation case",
    )


async def _verify_shortage_for(
    db: Postgres,
    relation: Relation,
    entities_by_id: Mapping[str, Entity],
) -> VerificationResult:
    source = entities_by_id[relation.source_entity_id]
    target = entities_by_id[relation.target_entity_id]
    if (
        EntityType(source.entity_type) == EntityType.SHORTAGE_AREA
        and EntityType(target.entity_type) == EntityType.COUNTY
    ):
        shortage_entity, county_entity = source, target
    elif (
        EntityType(source.entity_type) == EntityType.COUNTY
        and EntityType(target.entity_type) == EntityType.SHORTAGE_AREA
    ):
        shortage_entity, county_entity = target, source
    else:
        return _result(
            relation.relation_id,
            VerificationStatus.INFERRED,
            relation.confidence,
            "not_applicable",
            None,
            "SHORTAGE_FOR relation has no direct rule for this entity pair",
        )

    row = await db.fetchrow(
        """
        SELECT record_id, county_fips
        FROM workforce_shortage_records
        WHERE record_id = $1 AND county_fips = $2
        LIMIT 1
        """,
        shortage_entity.source_id,
        county_entity.source_id,
    )
    return _junction_result(relation, row, "workforce_shortage_records.record_id/county_fips")


RELATION_VERIFIERS: dict[RelationType, RelationVerifier] = {
    RelationType.MAPPED_TO: _verify_mapped_to,
    RelationType.ADDRESSES: _verify_addresses,
    RelationType.AUTHORED_BY: _verify_authored_by,
    RelationType.OFFERED_AT: _verify_offered_at,
    RelationType.SUPPORTS: _verify_supports,
    RelationType.TRAINED_FOR: _verify_trained_for,
    RelationType.SHORTAGE_FOR: _verify_shortage_for,
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


def _coerce_key(source_id: str, entity_type: EntityType) -> object | None:
    if entity_type == EntityType.INSTITUTION:
        try:
            return int(source_id)
        except (TypeError, ValueError):
            return None
    return source_id
