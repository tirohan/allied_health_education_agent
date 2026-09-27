"""Persist each mind-map run and its per-claim verification results.

Source for docs/ieee_bigdata_2026/export_live_verification_audit.py. Writes are
best-effort: a failure is logged and never breaks the user's request.
"""
from __future__ import annotations

from typing import Any

import structlog

from backend.app.agents.state import MindMapState
from backend.app.db.postgres import Postgres

logger = structlog.get_logger(__name__)


def claim_rows(state: MindMapState) -> list[tuple[Any, ...]]:
    """(entity_or_relation_id, kind, claim_text, status, method, source, snippet) per result."""
    entities = {e.entity_id: e for e in state.get("extracted_entities", [])}
    relations = {r.relation_id: r for r in state.get("extracted_relations", [])}
    rows = []
    for result in state.get("verification_results", []):
        rid = result.entity_or_relation_id
        if rid in relations:
            rel = relations[rid]
            src, tgt = entities.get(rel.source_entity_id), entities.get(rel.target_entity_id)
            kind = "relation"
            text = (
                f"{src.label if src else rel.source_entity_id} {rel.relation_type} "
                f"{tgt.label if tgt else rel.target_entity_id}"
            )
        else:
            ent = entities.get(rid)
            kind = "entity"
            text = f"{ent.entity_type} '{ent.label}' ({ent.source_table}:{ent.source_id})" if ent else rid
        rows.append((
            rid, kind, text[:1000], str(result.verification_status), result.verification_method,
            result.evidence_source, (result.evidence_snippet or "")[:1000] or None,
        ))
    return rows


async def persist_run(db: Postgres, state: MindMapState) -> int | None:
    graph = state.get("mindmap_graph")
    try:
        async with db.transaction() as conn:
            run_id = await conn.fetchval(
                """
                INSERT INTO mindmap_runs (query, snapshot_id, extraction_mode, retrieval_mode, n_nodes, n_edges)
                VALUES ($1, $2, $3, $4, $5, $6) RETURNING run_id
                """,
                state.get("query"),
                graph.snapshot_id if graph else state.get("snapshot_id"),
                state.get("extraction_mode"),
                state.get("retrieval_mode"),
                len(graph.nodes) if graph else None,
                len(graph.edges) if graph else None,
            )
            rows = claim_rows(state)
            if rows:
                await conn.executemany(
                    """
                    INSERT INTO verification_results (
                        run_id, entity_or_relation_id, claim_kind, claim_text,
                        verification_status, verification_method, evidence_source, evidence_snippet
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                    [(run_id, *row) for row in rows],
                )
        return run_id
    except Exception as exc:  # noqa: BLE001 - audit storage must never break a map request
        logger.warning("mindmap_run_persist_failed", error=str(exc))
        return None
