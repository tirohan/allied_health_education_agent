"""Regression tests for the 2026-09 reviewer revision (scope notes, filter, critic, provenance)."""

import importlib.util
import re
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from backend.app.agents.critic import critic_node
from backend.app.agents.mindmap import mindmap_node
from backend.app.agents.orchestrator import orchestrator_node, states_named
from backend.app.agents.state import (
    Relation,
    RelationType,
    RetrievalCollection,
    VerificationResult,
    VerificationStatus,
)
from backend.app.services.provenance import record_evidence
from test_critic import _config, _edge, _services, _state

DOCS = Path(__file__).resolve().parents[3] / "docs" / "ieee_bigdata_2026"


# ── Out-of-Georgia queries ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ohio_query_gets_scope_note_and_no_georgia_county_layer() -> None:
    state = await orchestrator_node(
        {
            "query": "rural physical therapy workforce shortage counties in Ohio",
            "collections": [
                RetrievalCollection.PAPERS,
                RetrievalCollection.RESOURCES,
                RetrievalCollection.PROGRAMS,
                RetrievalCollection.COMMUNITIES,
            ],
            "filters": {},
            "agent_trace": [],
        }
    )
    assert RetrievalCollection.COMMUNITIES not in state["collections"]
    assert RetrievalCollection.PROGRAMS in state["collections"]
    assert state["filters"].get("state") != "GA"
    assert len(state["scope_notes"]) == 1
    assert "Georgia only" in state["scope_notes"][0]
    assert "OH" in state["scope_notes"][0]


@pytest.mark.asyncio
async def test_georgia_query_has_no_scope_note() -> None:
    state = await orchestrator_node(
        {"query": "rural Georgia shortage counties", "filters": {}, "agent_trace": []}
    )
    assert state["scope_notes"] == []
    assert RetrievalCollection.COMMUNITIES in state["collections"]
    assert state["filters"]["state"] == "GA"


def test_states_named_prefers_longest_match() -> None:
    assert states_named("programs in West Virginia and Ohio") == ["WV", "OH"]
    assert states_named("opioid education for nurses") == []


@pytest.mark.asyncio
async def test_scope_note_and_snapshot_reach_graph_payload() -> None:
    state = await mindmap_node(
        {"query": "q", "scope_notes": ["note"], "snapshot_id": "7", "extracted_entities": []}
    )
    assert state["mindmap_graph"].scope_notes == ["note"]
    assert state["mindmap_graph"].snapshot_id == "7"
    unsnap = await mindmap_node({"query": "q", "extracted_entities": []})
    assert unsnap["mindmap_graph"].snapshot_id == "unsnapshotted"


# ── Semantic filter is diagnostic only ────────────────────────────────────────


def _load_apply_semantic_filter():
    sys.path.insert(0, str(DOCS))
    spec = importlib.util.spec_from_file_location("apply_semantic_filter", DOCS / "apply_semantic_filter.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def test_filter_keep_leaves_status_unreviewed() -> None:
    module = _load_apply_semantic_filter()
    with patch.object(module, "combined_filter", return_value=True):
        out = module.diagnostic_flag("Simulation in nursing education", "Nursing education", "unreviewed")
    assert out["passes_filter"] is True
    assert out["topic_validation_status"] == "unreviewed"


def test_filter_code_never_writes_validation_status() -> None:
    for name in ("apply_semantic_filter.py", "semantic_mapping_filter.py"):
        source = (DOCS / name).read_text()
        assert not re.search(r"UPDATE\s+paper_topics", source, re.IGNORECASE), name
        assert not re.search(r"SET\s+topic_validation_status", source, re.IGNORECASE), name


# ── Critic can only downgrade ────────────────────────────────────────────────


def _single_edge_state() -> dict:
    return _state(
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


@pytest.mark.asyncio
async def test_critic_supports_verdict_never_promotes_to_confirmed() -> None:
    canned = {"r1": {"verdict": "supports", "rationale": "Strong support."}}
    with patch("backend.app.agents.critic._critique", new_callable=AsyncMock, return_value=canned):
        result = await critic_node(_single_edge_state(), _config(_services()))
    statuses = {r.verification_status for r in result["verification_results"]}
    assert statuses == {VerificationStatus.INFERRED}


@pytest.mark.asyncio
async def test_critic_contradicts_sets_contested_on_edge_and_result() -> None:
    canned = {"r1": {"verdict": "contradicts", "rationale": "No."}}
    with patch("backend.app.agents.critic._critique", new_callable=AsyncMock, return_value=canned):
        result = await critic_node(_single_edge_state(), _config(_services()))
    assert result["mindmap_graph"].edges[0].verification_status == VerificationStatus.CONTESTED
    assert result["verification_results"][0].verification_status == VerificationStatus.CONTESTED


# ── Provenance endpoint ──────────────────────────────────────────────────────


class _FakeDb:
    def __init__(self, snapshot: int | None, provenance: list[dict], review: dict | None,
                 pk: str | None = "42") -> None:
        self.snapshot, self.provenance, self.review, self.pk = snapshot, provenance, review, pk
        self.fetch_calls: list[tuple] = []

    async def fetchrow(self, sql: str, *args: Any) -> dict | None:
        if "FROM snapshots" in sql:
            return {"snapshot_id": self.snapshot}
        if "FROM institutions" in sql:
            return {"pk": self.pk} if self.pk else None
        if "FROM verification_logs" in sql:
            return self.review
        raise AssertionError(sql)

    async def fetch(self, sql: str, *args: Any) -> list[dict]:
        assert "FROM field_provenance" in sql and "data_sources" not in sql
        self.fetch_calls.append(args)
        return self.provenance


@pytest.mark.asyncio
async def test_missing_provenance_returns_attributed_false() -> None:
    db = _FakeDb(snapshot=2, provenance=[], review=None)
    out = await record_evidence(db, "institutions", "100654", id_column="unitid")  # type: ignore[arg-type]
    assert out["attributed"] is False
    assert out["provenance"] == []
    assert out["snapshot_id"] == "2"
    assert db.fetch_calls == [(2, "institutions", "42")]  # unitid translated to the PK


@pytest.mark.asyncio
async def test_no_snapshot_is_unsnapshotted_and_unattributed() -> None:
    db = _FakeDb(snapshot=None, provenance=[{"column_name": "x"}], review=None)
    out = await record_evidence(db, "institutions", "42")  # type: ignore[arg-type]
    assert out["snapshot_id"] == "unsnapshotted"
    assert out["attributed"] is False
    assert db.fetch_calls == []


@pytest.mark.asyncio
async def test_provenance_rows_and_faculty_review_are_returned() -> None:
    rows = [{"column_name": "admission_rate", "source": "College Scorecard",
             "conflict_action": "overwritten", "batch_id": "b", "loaded_at": None}]
    review = {"verification_status": "CONFIRMED", "verified_by": "pt_faculty",
              "verified_date": "2026-09-26", "notes": "ok"}
    out = await record_evidence(_FakeDb(3, rows, review), "institutions", "42")  # type: ignore[arg-type]
    assert out["attributed"] is True
    assert out["provenance"] == rows
    assert out["faculty_review"]["verified_by"] == "pt_faculty"


@pytest.mark.asyncio
async def test_unsupported_id_column_is_rejected() -> None:
    with pytest.raises(ValueError):
        await record_evidence(_FakeDb(1, [], None), "research_papers", "x", id_column="doi")  # type: ignore[arg-type]


# ── Section 3: collection gating, extraction pruning, run store ──────────────

from backend.app.agents.extraction import _deterministic_extract  # noqa: E402
from backend.app.agents.state import Entity, RetrievedDoc  # noqa: E402
from backend.app.services.run_store import claim_rows  # noqa: E402


@pytest.mark.asyncio
async def test_non_geographic_query_drops_county_and_program_collections() -> None:
    state = await orchestrator_node({
        "query": "diagnostic medical sonography simulation-based training",
        "collections": [RetrievalCollection.PAPERS, RetrievalCollection.RESOURCES,
                        RetrievalCollection.PROGRAMS, RetrievalCollection.COMMUNITIES],
        "filters": {}, "agent_trace": [],
    })
    assert RetrievalCollection.COMMUNITIES not in state["collections"]
    assert RetrievalCollection.PROGRAMS not in state["collections"]
    assert RetrievalCollection.SIMULATION_CASES in state["collections"]


@pytest.mark.asyncio
async def test_program_question_keeps_programs() -> None:
    state = await orchestrator_node({
        "query": "which sonography degree programs are accredited",
        "collections": [RetrievalCollection.PAPERS, RetrievalCollection.PROGRAMS,
                        RetrievalCollection.COMMUNITIES],
        "filters": {}, "agent_trace": [],
    })
    assert RetrievalCollection.PROGRAMS in state["collections"]
    assert RetrievalCollection.COMMUNITIES not in state["collections"]


def _doc(i: str, collection: RetrievalCollection, title: str, text: str = "") -> RetrievedDoc:
    table = {RetrievalCollection.PAPERS: "research_papers", RetrievalCollection.RESOURCES: "resources",
             RetrievalCollection.SIMULATION_CASES: "simulation_cases",
             RetrievalCollection.COMMUNITIES: "county_profiles"}[collection]
    return RetrievedDoc(id=i, collection=collection, source_table=table, source_id=i,
                        title=title, text=text, score=0.5)


def test_extraction_dedupes_titles_and_drops_off_topic_docs() -> None:
    docs = [
        _doc("s1", RetrievalCollection.SIMULATION_CASES, "Opioid Overdose Simulation for Nursing Students"),
        _doc("r1", RetrievalCollection.RESOURCES, "Opioid overdose simulation for nursing students."),
        _doc("p1", RetrievalCollection.PAPERS, "Translational Educational Research"),
        _doc("c1", RetrievalCollection.COMMUNITIES, "Fulton County, GA"),
    ]
    entities, relations = _deterministic_extract(
        {"query": "opioid overdose simulation in rural Georgia", "retrieved_docs": docs}
    )
    ids = {e.source_id for e in entities}
    assert "s1" in ids and "r1" not in ids   # duplicate title kept once
    assert "p1" not in ids                   # no query topic words
    assert "c1" in ids                       # counties are exempt from the text gate
    assert len(relations) == len(entities) - 1


def test_claim_rows_cover_entities_and_relations() -> None:
    paper = Entity(entity_id="paper:1", entity_type="Paper", label="P", source_table="research_papers",
                   source_id="1", confidence=0.8)
    topic = Entity(entity_id="query_root", entity_type="Topic", label="T", source_table="topic_modules",
                   source_id="t", confidence=0.8)
    rel = Relation(relation_id="r1", source_entity_id="paper:1", target_entity_id="query_root",
                   relation_type=RelationType.SUPPORTS, confidence=0.7)
    results = [
        VerificationResult(entity_or_relation_id="paper:1", verification_status=VerificationStatus.CONFIRMED,
                           verification_method="direct_lookup", evidence_source="research_papers"),
        VerificationResult(entity_or_relation_id="r1", verification_status=VerificationStatus.INFERRED,
                           verification_method="text_match", evidence_snippet="x"),
    ]
    rows = claim_rows({"extracted_entities": [paper, topic], "extracted_relations": [rel],
                       "verification_results": results})
    assert [(r[0], r[1], r[3]) for r in rows] == [("paper:1", "entity", "CONFIRMED"), ("r1", "relation", "INFERRED")]
    assert rows[1][2] == "P SUPPORTS T"


# ── Query-root topic tag no longer defaults to a wrong specific category ─────

from backend.app.agents.extraction import _topic_id_for_query  # noqa: E402


def test_topic_id_matches_specific_categories() -> None:
    assert _topic_id_for_query("physical therapy workforce in Ohio") == "rehabilitation_sciences"
    assert _topic_id_for_query("diagnostic medical sonography simulation-based training") == "diagnostic_imaging"
    assert _topic_id_for_query("point-of-care ultrasound education for clinicians") == "diagnostic_imaging"
    assert _topic_id_for_query("radiologic technology program accreditation standards") == "diagnostic_imaging"
    assert _topic_id_for_query("Which allied health programs are available near Georgia "
                               "health professional shortage counties?") == "workforce_development"
    assert _topic_id_for_query("opioid education in rural Georgia") == "opioid_substance_use"


def test_topic_id_falls_back_to_generic_not_a_specific_wrong_category() -> None:
    # Old behavior silently mislabeled any unmatched query as "behavioral_health_substance_use".
    assert _topic_id_for_query("physician assistant surgical skills training") == "allied_health_education"


# ── Topic-words-in-blob helper (mapped_to / relevant_to fallback) ────────────

from backend.app.agents.verification import _topic_words_in_text  # noqa: E402


def test_topic_words_in_text_requires_a_real_shared_word() -> None:
    assert _topic_words_in_text("Opioid and Substance Use Education", "Opioid Overdose Toolkit") is True
    assert _topic_words_in_text("Health Workforce Development", "Nursing Roles and Practices") is False
    assert _topic_words_in_text("", "anything") is False
