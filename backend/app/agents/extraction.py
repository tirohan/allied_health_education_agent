import json
import re
from typing import Any

from langchain_core.runnables import RunnableConfig

from backend.app.agents.state import (
    AgentStep,
    Entity,
    EntityType,
    MindMapState,
    Relation,
    RelationType,
    RetrievalCollection,
)

COLLECTION_ENTITY: dict[RetrievalCollection, tuple[EntityType, RelationType]] = {
    RetrievalCollection.PAPERS: (EntityType.PAPER, RelationType.SUPPORTS),
    RetrievalCollection.RESOURCES: (EntityType.RESOURCE, RelationType.MAPPED_TO),
    RetrievalCollection.PROGRAMS: (EntityType.PROGRAM, RelationType.RELEVANT_TO),
    RetrievalCollection.COMMUNITIES: (EntityType.COUNTY, RelationType.LOCATED_IN),
    RetrievalCollection.SIMULATION_CASES: (EntityType.SIMULATION_CASE, RelationType.TRAINED_FOR),
}


async def extraction_node(
    state: MindMapState,
    config: RunnableConfig | None = None,
) -> MindMapState:
    configurable = (config or {}).get("configurable", {})
    services = configurable.get("services") if isinstance(configurable, dict) else None
    entities, relations = _deterministic_extract(state)
    requested_mode = str(state.get("extraction_mode") or "auto").strip().lower()
    extraction_mode = "deterministic"

    use_llm = requested_mode in {"auto", "llm"}
    if requested_mode == "deterministic":
        use_llm = False

    if (
        use_llm
        and services is not None
        and getattr(services, "settings", None) is not None
    ):
        api_key = services.settings.openai_api_key
        secret = api_key.get_secret_value().strip() if api_key is not None else ""
        if secret:
            try:
                llm_entities, llm_relations = await _llm_extract(
                    state,
                    secret,
                    services.settings.openai_model,
                )
                if llm_entities:
                    if requested_mode == "llm":
                        entities, relations = llm_entities, llm_relations
                        extraction_mode = "llm_structured"
                    else:
                        entities, relations = _merge_extractions(
                            entities,
                            relations,
                            llm_entities,
                            llm_relations,
                        )
                        extraction_mode = "llm_structured"
            except Exception as exc:  # noqa: BLE001 - fall back to deterministic path
                extraction_mode = f"deterministic_fallback:{exc.__class__.__name__}"
        elif requested_mode == "llm":
            extraction_mode = "deterministic_fallback:MissingOpenAIKey"

    return {
        **state,
        "extracted_entities": entities,
        "extracted_relations": relations,
        "agent_trace": [
            *state.get("agent_trace", []),
            AgentStep(
                agent="extraction",
                message=(
                    f"Extracted {len(entities)} entities and {len(relations)} relations "
                    f"using {extraction_mode}."
                ),
                metadata={"mode": extraction_mode},
            ),
        ],
    }


def _deterministic_extract(state: MindMapState) -> tuple[list[Entity], list[Relation]]:
    entities: list[Entity] = []
    relations: list[Relation] = []
    query = state.get("refined_query") or state["query"]
    topic_id = _topic_id_for_query(query)
    query_entity = Entity(
        entity_id="query_root",
        entity_type=EntityType.TOPIC,
        label=query,
        summary="Root concept generated from the user query.",
        source_table="topic_modules",
        source_id=topic_id,
        confidence=0.8,
    )
    entities.append(query_entity)

    for doc in state.get("retrieved_docs", []):
        mapping = COLLECTION_ENTITY.get(RetrievalCollection(doc.collection))
        if mapping is None:
            continue
        entity_type, relation_type = mapping
        # County relations point from root topic to county for clearer navigation.
        if entity_type == EntityType.COUNTY:
            source_id = query_entity.entity_id
            target_id = f"{entity_type.value.lower()}:{doc.source_id}"
            entity_id = target_id
            relation_source = source_id
            relation_target = target_id
        else:
            entity_id = f"{entity_type.value.lower()}:{doc.source_id}"
            relation_source = entity_id
            relation_target = query_entity.entity_id

        entity = Entity(
            entity_id=entity_id,
            entity_type=entity_type,
            label=doc.title[:160],
            summary=doc.text[:500],
            source_table=doc.source_table,
            source_id=doc.source_id,
            confidence=min(0.95, max(0.55, float(doc.score) if doc.score > 1 else float(doc.score) * 8 + 0.4)),
            evidence_doc_ids=[doc.id],
        )
        if entity.entity_id not in {item.entity_id for item in entities}:
            entities.append(entity)
        relations.append(
            Relation(
                relation_id=f"{relation_source}->{relation_target}:{relation_type.value}",
                source_entity_id=relation_source,
                target_entity_id=relation_target,
                relation_type=relation_type,
                evidence_text=doc.text[:500],
                confidence=entity.confidence,
                evidence_doc_ids=[doc.id],
            )
        )
    return entities, relations


async def _llm_extract(
    state: MindMapState,
    api_key: str,
    model: str,
) -> tuple[list[Entity], list[Relation]]:
    from openai import AsyncOpenAI

    docs = state.get("retrieved_docs", [])[:20]
    payload = [
        {
            "id": doc.id,
            "collection": str(doc.collection),
            "source_table": doc.source_table,
            "source_id": doc.source_id,
            "title": doc.title,
            "text": doc.text[:800],
        }
        for doc in docs
    ]
    allowed_entity_types = ", ".join(item.value for item in EntityType)
    allowed_relation_types = ", ".join(item.value for item in RelationType)
    prompt = (
        "Extract entities and relations for an allied health education mind map. "
        "Return JSON with keys entities and relations. "
        "Each entity needs entity_id, entity_type, label, summary, source_table, "
        "source_id, confidence. "
        "Each relation needs relation_id, source_entity_id, target_entity_id, "
        "relation_type, evidence_text, confidence. "
        f"entity_type must be one of: {allowed_entity_types}. "
        f"relation_type must be one of: {allowed_relation_types}. "
        "Only use source_table and source_id values from the provided documents. "
        "Include a Topic entity for the query and link documents to it. "
        f"Query: {state.get('refined_query') or state['query']}. "
        f"Documents: {json.dumps(payload)}"
    )
    client = AsyncOpenAI(api_key=api_key)
    response = await client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You extract grounded entities and relations as strict JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )
    content = response.choices[0].message.content or "{}"
    data = json.loads(content)
    entities = [
        entity
        for item in data.get("entities", [])
        if (entity := _coerce_entity(item)) is not None
    ]
    relations = [
        relation
        for item in data.get("relations", [])
        if (relation := _coerce_relation(item)) is not None
    ]
    if not entities:
        raise ValueError("LLM returned no valid entities")
    return entities, relations


def _merge_extractions(
    deterministic_entities: list[Entity],
    deterministic_relations: list[Relation],
    llm_entities: list[Entity],
    llm_relations: list[Relation],
) -> tuple[list[Entity], list[Relation]]:
    """Prefer LLM labels/summaries, but keep grounded deterministic document nodes."""

    by_id: dict[str, Entity] = {entity.entity_id: entity for entity in deterministic_entities}
    for entity in llm_entities:
        existing = by_id.get(entity.entity_id)
        if existing is None:
            by_id[entity.entity_id] = entity
            continue
        by_id[entity.entity_id] = existing.model_copy(
            update={
                "label": entity.label or existing.label,
                "summary": entity.summary or existing.summary,
                "confidence": max(existing.confidence, entity.confidence),
                "evidence_doc_ids": list(
                    dict.fromkeys([*existing.evidence_doc_ids, *entity.evidence_doc_ids])
                ),
            }
        )

    relation_by_id: dict[str, Relation] = {
        relation.relation_id: relation for relation in deterministic_relations
    }
    for relation in llm_relations:
        relation_by_id[relation.relation_id] = relation
    return list(by_id.values()), list(relation_by_id.values())


_ENTITY_ALIASES = {
    "topic": EntityType.TOPIC,
    "competency": EntityType.COMPETENCY,
    "discipline": EntityType.DISCIPLINE,
    "paper": EntityType.PAPER,
    "research_paper": EntityType.PAPER,
    "article": EntityType.PAPER,
    "resource": EntityType.RESOURCE,
    "oer": EntityType.RESOURCE,
    "program": EntityType.PROGRAM,
    "institution": EntityType.INSTITUTION,
    "county": EntityType.COUNTY,
    "community": EntityType.COUNTY,
    "shortagearea": EntityType.SHORTAGE_AREA,
    "shortage_area": EntityType.SHORTAGE_AREA,
    "simulationcase": EntityType.SIMULATION_CASE,
    "simulation_case": EntityType.SIMULATION_CASE,
    "simulation": EntityType.SIMULATION_CASE,
    "author": EntityType.AUTHOR,
}

_RELATION_ALIASES = {
    "supports": RelationType.SUPPORTS,
    "addresses": RelationType.ADDRESSES,
    "offered_at": RelationType.OFFERED_AT,
    "accredited_by": RelationType.ACCREDITED_BY,
    "located_in": RelationType.LOCATED_IN,
    "shortage_for": RelationType.SHORTAGE_FOR,
    "mapped_to": RelationType.MAPPED_TO,
    "authored_by": RelationType.AUTHORED_BY,
    "trained_for": RelationType.TRAINED_FOR,
    "relevant_to": RelationType.RELEVANT_TO,
    "related_to": RelationType.RELEVANT_TO,
}


def _normalize_entity_type(value: Any) -> EntityType | None:
    if isinstance(value, EntityType):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    for item in EntityType:
        if text == item.value or text.lower() == item.value.lower():
            return item
    return _ENTITY_ALIASES.get(text.lower().replace(" ", "_"))


def _normalize_relation_type(value: Any) -> RelationType | None:
    if isinstance(value, RelationType):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    for item in RelationType:
        if text == item.value or text.lower() == item.value.lower():
            return item
    return _RELATION_ALIASES.get(text.lower().replace(" ", "_"))


def _clamp_confidence(value: Any, default: float = 0.5) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, score))


def _coerce_entity(item: dict[str, Any]) -> Entity | None:
    if not isinstance(item, dict):
        return None
    entity_type = _normalize_entity_type(item.get("entity_type"))
    entity_id = str(item.get("entity_id") or item.get("id") or "").strip()
    label = str(item.get("label") or item.get("title") or "").strip()
    source_table = str(item.get("source_table") or "").strip()
    source_id = str(item.get("source_id") or "").strip()
    if not entity_type or not entity_id or not label or not source_table or not source_id:
        return None
    try:
        return Entity(
            entity_id=entity_id,
            entity_type=entity_type,
            label=label,
            summary=str(item.get("summary") or "")[:1000],
            source_table=source_table,
            source_id=source_id,
            confidence=_clamp_confidence(item.get("confidence"), 0.55),
            evidence_doc_ids=[
                str(doc_id)
                for doc_id in item.get("evidence_doc_ids", [])
                if doc_id is not None
            ],
        )
    except Exception:  # noqa: BLE001
        return None


def _coerce_relation(item: dict[str, Any]) -> Relation | None:
    if not isinstance(item, dict):
        return None
    relation_type = _normalize_relation_type(item.get("relation_type") or item.get("label"))
    source_entity_id = str(
        item.get("source_entity_id") or item.get("source_id") or item.get("source") or ""
    ).strip()
    target_entity_id = str(
        item.get("target_entity_id") or item.get("target_id") or item.get("target") or ""
    ).strip()
    if not relation_type or not source_entity_id or not target_entity_id:
        return None
    relation_id = str(
        item.get("relation_id")
        or f"{source_entity_id}:{relation_type.value}:{target_entity_id}"
    )
    try:
        return Relation(
            relation_id=relation_id,
            source_entity_id=source_entity_id,
            target_entity_id=target_entity_id,
            relation_type=relation_type,
            evidence_text=str(item.get("evidence_text") or "")[:1000],
            confidence=_clamp_confidence(item.get("confidence"), 0.5),
            evidence_doc_ids=[
                str(doc_id)
                for doc_id in item.get("evidence_doc_ids", [])
                if doc_id is not None
            ],
        )
    except Exception:  # noqa: BLE001
        return None


def _topic_id_for_query(query: str) -> str:
    lowered = query.lower()
    if re.search(r"opioid|substance", lowered):
        return "opioid_substance_use"
    if re.search(r"fall|aging", lowered):
        return "aging_fall_prevention"
    if re.search(r"digital|ai literacy|telehealth", lowered):
        return "digital_health_ai_literacy"
    if re.search(r"chronic|diabetes", lowered):
        return "chronic_disease"
    return "behavioral_health_substance_use"
