"""Column provenance and faculty-review lookups for the evidence page.

Provenance comes only from field_provenance rows of the latest complete snapshot.
Nothing is inferred from institutions.data_sources: a record with no rows is
returned with attributed=False.
"""
from __future__ import annotations

from typing import Any

import asyncpg

from backend.app.db.postgres import Postgres

UNSNAPSHOTTED = "unsnapshotted"
# Alternate keys the app uses for a table, mapped to the field_provenance record_id
# (the table's primary key). Institution cards carry the IPEDS unitid.
_ALT_KEYS: dict[tuple[str, str], str] = {
    ("institutions", "unitid"): (
        "SELECT institution_id::text AS pk FROM institutions WHERE unitid::text = $1"
    ),
}


async def latest_snapshot_id(db: Postgres) -> int | None:
    try:
        row = await db.fetchrow(
            "SELECT max(snapshot_id) AS snapshot_id FROM snapshots WHERE status = 'complete'"
        )
    except asyncpg.UndefinedTableError:  # database predates the snapshots migration
        return None
    return row["snapshot_id"] if row else None


async def record_evidence(
    db: Postgres,
    table_name: str,
    record_id: str,
    id_column: str | None = None,
) -> dict[str, Any]:
    if id_column and (table_name, id_column) not in _ALT_KEYS:
        raise ValueError(f"Unsupported id_column {id_column!r} for table {table_name!r}")

    snapshot_id = await latest_snapshot_id(db)
    pk: str | None = record_id
    if id_column:
        row = await db.fetchrow(_ALT_KEYS[(table_name, id_column)], record_id)
        pk = row["pk"] if row else None

    provenance: list[dict[str, Any]] = []
    if snapshot_id is not None and pk is not None:
        provenance = [
            dict(r)
            for r in await db.fetch(
                """
                SELECT column_name, source, conflict_action, batch_id, loaded_at
                FROM field_provenance
                WHERE snapshot_id = $1 AND table_name = $2 AND record_id = $3
                ORDER BY column_name, source
                """,
                snapshot_id,
                table_name,
                pk,
            )
        ]

    # Same store submit_faculty_review writes and _apply_faculty_overrides rereads.
    review = await db.fetchrow(
        """
        SELECT verification_status, verified_by, verified_date, notes
        FROM verification_logs
        WHERE evidence_level = 'faculty_review' AND record_type = $1 AND record_id = $2
        ORDER BY verified_date DESC, verification_id DESC
        LIMIT 1
        """,
        table_name,
        record_id,
    )
    return {
        "table_name": table_name,
        "record_id": record_id,
        "snapshot_id": str(snapshot_id) if snapshot_id is not None else UNSNAPSHOTTED,
        "attributed": bool(provenance),
        "provenance": provenance,
        "faculty_review": dict(review) if review else None,
    }
