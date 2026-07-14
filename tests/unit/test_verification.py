from typing import Any

import pytest

from backend.app.agents.state import Entity, EntityType, VerificationStatus
from backend.app.agents.verification import verify_entity


class FakePostgres:
    def __init__(self, row: dict[str, Any] | None) -> None:
        self.row = row

    async def fetchrow(self, _sql: str, *_args: object) -> dict[str, Any] | None:
        return self.row


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
