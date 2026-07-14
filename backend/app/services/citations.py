from collections.abc import Mapping
from typing import Any

from backend.app.agents.state import Citation


def citation_from_record(
    source_table: str,
    source_id: str,
    record: Mapping[str, Any],
    snippet_field: str | None = None,
) -> Citation:
    title = record.get("title") or record.get("topic_label") or record.get("competency_name")
    title = title or record.get("institution_name") or record.get("county_name") or source_id
    snippet = str(record.get(snippet_field, ""))[:500] if snippet_field else None
    url = record.get("url") or record.get("landing_page_url") or record.get("source_url")
    return Citation(
        source_table=source_table,
        source_id=source_id,
        label=str(title),
        url=str(url) if url else None,
        doi=str(record["doi"]) if record.get("doi") else None,
        evidence_snippet=snippet or None,
    )
