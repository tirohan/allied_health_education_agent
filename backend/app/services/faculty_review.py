from __future__ import annotations

from datetime import date
from typing import Any, Literal

from backend.app.db.postgres import Postgres

FacultyDecision = Literal["Useful", "Not relevant", "Needs review"]


async def submit_faculty_review(
    db: Postgres,
    *,
    record_type: str,
    record_id: str,
    record_title: str | None,
    decision: FacultyDecision,
    reviewer: str,
    notes: str | None = None,
) -> dict[str, Any]:
    """Persist faculty feedback into verification logs and the manual review queue."""

    status_map = {
        "Useful": "CONFIRMED",
        "Not relevant": "REFUTED",
        "Needs review": "UNVERIFIED",
    }
    verification_status = status_map[decision]
    note_text = notes or f"Faculty marked this item as {decision}."

    await db.execute(
        """
        INSERT INTO verification_logs (
            record_type, record_id, verification_status, evidence_level,
            verified_by, verified_date, notes, next_action
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """,
        record_type,
        record_id,
        verification_status,
        "faculty_review",
        reviewer,
        date.today(),
        note_text,
        decision,
    )

    queue_id = None
    if decision == "Needs review":
        row = await db.fetchrow(
            """
            INSERT INTO manual_review_queue (
                record_type, record_id, record_title, priority_score,
                review_flags, workflow_status, assigned_reviewer, notes
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING queue_id
            """,
            record_type,
            record_id,
            record_title,
            3,
            "faculty_flagged",
            "Queued",
            reviewer,
            note_text,
        )
        queue_id = row["queue_id"] if row else None

    return {
        "status": "saved",
        "decision": decision,
        "verification_status": verification_status,
        "queue_id": queue_id,
        "message": (
            "Thank you. Your review helps improve trust and future recommendations."
        ),
    }
