from typing import Any

import pytest

from backend.app.agents.state import Entity, EntityType, Relation, RelationType, VerificationStatus
from backend.app.agents.verification import (
    _apply_faculty_overrides,
    _topic_overlap,
    verify_entity,
    verify_relation,
)


class FakePostgres:
    def __init__(self, row: dict[str, Any] | None) -> None:
        self.row = row

    async def fetchrow(self, _sql: str, *_args: object) -> dict[str, Any] | None:
        return self.row


class QueuedPostgres:
    """Returns canned results in call order -- for verifiers that issue more
    than one sequential query, where a single fixed response isn't enough."""

    def __init__(
        self,
        fetchrow_results: list[dict[str, Any] | None] | None = None,
        fetch_results: list[list[dict[str, Any]]] | None = None,
    ) -> None:
        self._fetchrow_results = list(fetchrow_results or [])
        self._fetch_results = list(fetch_results or [])

    async def fetchrow(self, _sql: str, *_args: object) -> dict[str, Any] | None:
        return self._fetchrow_results.pop(0) if self._fetchrow_results else None

    async def fetch(self, _sql: str, *_args: object) -> list[dict[str, Any]]:
        return self._fetch_results.pop(0) if self._fetch_results else []


@pytest.mark.asyncio
async def test_verify_entity_confirmed() -> None:
    entity = Entity(
        entity_id="paper:W1",
        entity_type=EntityType.PAPER,
        label="Simulation Training",
        source_table="research_papers",
        source_id="W1",
        confidence=0.9,
    )
    result = await verify_entity(FakePostgres({"paper_id": "W1", "title": "Simulation Training"}), entity)  # type: ignore[arg-type]
    assert result.verification_status == VerificationStatus.CONFIRMED


@pytest.mark.asyncio
async def test_verify_entity_refuted_when_record_missing() -> None:
    entity = Entity(
        entity_id="paper:missing",
        entity_type=EntityType.PAPER,
        label="Missing",
        source_table="research_papers",
        source_id="missing",
        confidence=0.9,
    )
    result = await verify_entity(FakePostgres(None), entity)  # type: ignore[arg-type]
    assert result.verification_status == VerificationStatus.REFUTED


def test_topic_overlap_ignores_shared_generic_words() -> None:
    # Regression test: live-verified against the real DB, "Interprofessional
    # Education" was falsely matching "Medical education" on the shared word
    # "education" alone before _topic_overlap replaced the generic overlap check.
    assert _topic_overlap("Interprofessional Education", "Medical education") is False


def test_topic_overlap_matches_real_variant() -> None:
    assert _topic_overlap(
        "Interprofessional Education", "Interprofessional Education and Collaboration"
    ) is True


def _entities(*items: Entity) -> dict[str, Entity]:
    return {item.entity_id: item for item in items}


@pytest.mark.asyncio
async def test_verify_supports_confirmed_via_paper_topics() -> None:
    paper = Entity(
        entity_id="paper:W1", entity_type=EntityType.PAPER, label="p",
        source_table="research_papers", source_id="W1", confidence=0.8,
    )
    topic = Entity(
        entity_id="query_root", entity_type=EntityType.TOPIC, label="q",
        source_table="topic_modules", source_id="interprofessional_education", confidence=0.8,
    )
    relation = Relation(
        relation_id="r1", source_entity_id="paper:W1", target_entity_id="query_root",
        relation_type=RelationType.SUPPORTS, confidence=0.7,
    )
    db = QueuedPostgres(
        fetchrow_results=[{"topic_label": "Interprofessional Education"}],
        fetch_results=[[{"topic_name": "Interprofessional Education and Collaboration"}]],
    )
    result = await verify_relation(db, relation, _entities(paper, topic))  # type: ignore[arg-type]
    assert result.verification_status == VerificationStatus.CONFIRMED
    assert result.verification_method == "text_match"


@pytest.mark.asyncio
async def test_verify_supports_inferred_when_topics_exist_but_dont_match() -> None:
    paper = Entity(
        entity_id="paper:W2", entity_type=EntityType.PAPER, label="p",
        source_table="research_papers", source_id="W2", confidence=0.8,
    )
    topic = Entity(
        entity_id="query_root", entity_type=EntityType.TOPIC, label="q",
        source_table="topic_modules", source_id="opioid_substance_use", confidence=0.8,
    )
    relation = Relation(
        relation_id="r2", source_entity_id="paper:W2", target_entity_id="query_root",
        relation_type=RelationType.SUPPORTS, confidence=0.7,
    )
    db = QueuedPostgres(
        fetchrow_results=[
            {"topic_label": "Opioid and Substance Use Education"},
            {"topic_tags_inferred": None},
        ],
        fetch_results=[[{"topic_name": "Electrical engineering"}]],
    )
    result = await verify_relation(db, relation, _entities(paper, topic))  # type: ignore[arg-type]
    assert result.verification_status == VerificationStatus.INFERRED


@pytest.mark.asyncio
async def test_verify_supports_unverified_when_no_topic_data() -> None:
    paper = Entity(
        entity_id="paper:W3", entity_type=EntityType.PAPER, label="p",
        source_table="research_papers", source_id="W3", confidence=0.8,
    )
    topic = Entity(
        entity_id="query_root", entity_type=EntityType.TOPIC, label="q",
        source_table="topic_modules", source_id="opioid_substance_use", confidence=0.8,
    )
    relation = Relation(
        relation_id="r3", source_entity_id="paper:W3", target_entity_id="query_root",
        relation_type=RelationType.SUPPORTS, confidence=0.7,
    )
    db = QueuedPostgres(
        fetchrow_results=[{"topic_label": "Opioid and Substance Use Education"}, None],
        fetch_results=[[]],
    )
    result = await verify_relation(db, relation, _entities(paper, topic))  # type: ignore[arg-type]
    assert result.verification_status == VerificationStatus.UNVERIFIED


@pytest.mark.asyncio
async def test_verify_trained_for_confirmed_via_case_topics_junction() -> None:
    case = Entity(
        entity_id="simulationcase:sc1", entity_type=EntityType.SIMULATION_CASE, label="c",
        source_table="simulation_cases", source_id="sc1", confidence=0.8,
    )
    topic = Entity(
        entity_id="query_root", entity_type=EntityType.TOPIC, label="q",
        source_table="topic_modules", source_id="opioid_substance_use", confidence=0.8,
    )
    relation = Relation(
        relation_id="r4", source_entity_id="simulationcase:sc1", target_entity_id="query_root",
        relation_type=RelationType.TRAINED_FOR, confidence=0.7,
    )
    db = QueuedPostgres(
        fetchrow_results=[
            {"topic_label": "Opioid and Substance Use Education"},
            {"simulation_case_id": "sc1", "topic_tag": "opioid_substance_use"},
        ],
    )
    result = await verify_relation(db, relation, _entities(case, topic))  # type: ignore[arg-type]
    assert result.verification_status == VerificationStatus.CONFIRMED
    assert result.verification_method == "junction_check"


@pytest.mark.asyncio
async def test_verify_trained_for_inferred_via_topic_tags_fallback() -> None:
    case = Entity(
        entity_id="simulationcase:sc2", entity_type=EntityType.SIMULATION_CASE, label="c",
        source_table="simulation_cases", source_id="sc2", confidence=0.8,
    )
    topic = Entity(
        entity_id="query_root", entity_type=EntityType.TOPIC, label="q",
        source_table="topic_modules", source_id="opioid_substance_use", confidence=0.8,
    )
    relation = Relation(
        relation_id="r5", source_entity_id="simulationcase:sc2", target_entity_id="query_root",
        relation_type=RelationType.TRAINED_FOR, confidence=0.7,
    )
    db = QueuedPostgres(
        fetchrow_results=[
            {"topic_label": "Opioid and Substance Use Education"},
            None,
            {"topic_tags": "opioid substance use disorder education"},
        ],
    )
    result = await verify_relation(db, relation, _entities(case, topic))  # type: ignore[arg-type]
    assert result.verification_status == VerificationStatus.INFERRED
    assert result.verification_method == "text_match"


@pytest.mark.asyncio
async def test_verify_shortage_for_confirmed() -> None:
    shortage = Entity(
        entity_id="shortagearea:s1", entity_type=EntityType.SHORTAGE_AREA, label="s",
        source_table="workforce_shortage_records", source_id="rec1", confidence=0.8,
    )
    county = Entity(
        entity_id="county:c1", entity_type=EntityType.COUNTY, label="c",
        source_table="county_profiles", source_id="13033", confidence=0.8,
    )
    relation = Relation(
        relation_id="r6", source_entity_id="shortagearea:s1", target_entity_id="county:c1",
        relation_type=RelationType.SHORTAGE_FOR, confidence=0.7,
    )
    db = FakePostgres({"record_id": "rec1", "county_fips": "13033"})
    result = await verify_relation(db, relation, _entities(shortage, county))  # type: ignore[arg-type]
    assert result.verification_status == VerificationStatus.CONFIRMED
    assert result.verification_method == "junction_check"


@pytest.mark.asyncio
async def test_verify_shortage_for_unverified_when_no_match() -> None:
    shortage = Entity(
        entity_id="shortagearea:s2", entity_type=EntityType.SHORTAGE_AREA, label="s",
        source_table="workforce_shortage_records", source_id="rec2", confidence=0.8,
    )
    county = Entity(
        entity_id="county:c2", entity_type=EntityType.COUNTY, label="c",
        source_table="county_profiles", source_id="99999", confidence=0.8,
    )
    relation = Relation(
        relation_id="r7", source_entity_id="shortagearea:s2", target_entity_id="county:c2",
        relation_type=RelationType.SHORTAGE_FOR, confidence=0.7,
    )
    db = FakePostgres(None)
    result = await verify_relation(db, relation, _entities(shortage, county))  # type: ignore[arg-type]
    assert result.verification_status == VerificationStatus.UNVERIFIED


class FaultyReviewPostgres:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    async def fetch(self, _sql: str, *_args: object) -> list[dict[str, Any]]:
        return self.rows


@pytest.mark.asyncio
async def test_apply_faculty_overrides_leaves_unreviewed_entities_alone() -> None:
    from backend.app.agents.verification import _result

    entity = Entity(
        entity_id="resource:r1", entity_type=EntityType.RESOURCE, label="r",
        source_table="resources", source_id="r1", confidence=0.8,
    )
    original = _result("resource:r1", VerificationStatus.CONFIRMED, 0.8, "direct_lookup", None, None)
    updated = await _apply_faculty_overrides(FaultyReviewPostgres([]), [original], [entity])  # type: ignore[arg-type]
    assert updated == [original]


@pytest.mark.asyncio
async def test_apply_faculty_overrides_refutes_not_relevant_items() -> None:
    from backend.app.agents.verification import _result

    entity = Entity(
        entity_id="resource:r2", entity_type=EntityType.RESOURCE, label="r",
        source_table="resources", source_id="r2", confidence=0.8,
    )
    original = _result("resource:r2", VerificationStatus.CONFIRMED, 0.8, "direct_lookup", None, None)
    db = FaultyReviewPostgres(
        [
            {
                "record_type": "resources",
                "record_id": "r2",
                "verification_status": "REFUTED",
                "verified_by": "faculty_user",
                "verified_date": "2026-07-01",
                "notes": "Outdated.",
            }
        ]
    )
    updated = await _apply_faculty_overrides(db, [original], [entity])  # type: ignore[arg-type]
    assert updated[0].verification_status == VerificationStatus.REFUTED
    assert updated[0].verification_method == "faculty_review"
