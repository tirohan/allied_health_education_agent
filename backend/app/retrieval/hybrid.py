import re
from collections import defaultdict
from typing import Any

from qdrant_client import models

from backend.app.agents.state import RetrievalCollection, RetrievedDoc
from backend.app.db.postgres import Postgres
from backend.app.db.qdrant import QdrantStore
from backend.app.retrieval.embedder import Embedder

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "what",
    "which",
    "with",
}


def keyword_tsquery(query: str) -> str:
    """Build an OR full-text query from salient tokens in a natural language question."""
    tokens = [
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", query.lower())
        if token not in _STOPWORDS
    ]
    # Prefer distinctive research terms first.
    preferred = [
        token
        for token in tokens
        if token
        in {
            "opioid",
            "opioids",
            "interprofessional",
            "ipe",
            "simulation",
            "shortage",
            "hpsa",
            "rural",
            "georgia",
            "substance",
            "behavioral",
        }
    ]
    chosen = preferred or tokens[:6]
    if not chosen:
        return query
    return " OR ".join(chosen)


class HybridSearch:
    def __init__(self, db: Postgres, qdrant: QdrantStore, embedder: Embedder) -> None:
        self.db = db
        self.qdrant = qdrant
        self.embedder = embedder

    async def search(
        self,
        query: str,
        collections: list[RetrievalCollection],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedDoc]:
        filters = filters or {}
        vector_results: list[RetrievedDoc] = []
        sql_results: list[RetrievedDoc] = []
        # Keep representation across collections instead of letting one domain dominate.
        per_collection = max(3, (top_k + len(collections) - 1) // max(len(collections), 1))
        sql_query = keyword_tsquery(query)
        for collection in collections:
            vector_results.extend(
                await self._vector_search(query, collection, per_collection, filters)
            )
            sql_results.extend(
                await self._sql_search(sql_query, collection, per_collection, filters)
            )
        return reciprocal_rank_fusion(vector_results, sql_results, top_k=top_k)

    async def _vector_search(
        self,
        query: str,
        collection: RetrievalCollection,
        top_k: int,
        filters: dict[str, Any],
    ) -> list[RetrievedDoc]:
        if not await self.qdrant.client.collection_exists(collection.value):
            return []
        query_vector = await self.embedder.embed(query)
        result = await self.qdrant.client.query_points(
            collection_name=collection.value,
            query=query_vector,
            limit=top_k,
            query_filter=_qdrant_filter(filters),
            with_payload=True,
        )
        docs: list[RetrievedDoc] = []
        for rank, point in enumerate(result.points, start=1):
            payload = dict(point.payload or {})
            docs.append(
                RetrievedDoc(
                    id=str(point.id),
                    collection=collection,
                    source_table=str(payload.get("source_table", "")),
                    source_id=str(payload.get("source_id", point.id)),
                    title=str(payload.get("title", point.id)),
                    text=str(payload.get("text", "")),
                    score=float(point.score or 0.0),
                    vector_rank=rank,
                    payload=payload,
                )
            )
        return docs

    async def _sql_search(
        self,
        query: str,
        collection: RetrievalCollection,
        top_k: int,
        filters: dict[str, Any],
    ) -> list[RetrievedDoc]:
        if collection == RetrievalCollection.PAPERS:
            return await self._sql_search_papers(query, top_k, filters)
        if collection == RetrievalCollection.RESOURCES:
            return await self._sql_search_resources(query, top_k, filters)
        if collection == RetrievalCollection.PROGRAMS:
            return await self._sql_search_programs(query, top_k, filters)
        if collection == RetrievalCollection.COMMUNITIES:
            return await self._sql_search_communities(query, top_k, filters)
        if collection == RetrievalCollection.SIMULATION_CASES:
            return await self._sql_search_simulation_cases(query, top_k, filters)
        return []

    async def _sql_search_papers(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any],
    ) -> list[RetrievedDoc]:
        rows = await self.db.fetch(
            """
            SELECT paper_id, title, abstract, publication_year, relevance_status,
                   source_database, doi,
                   ts_rank_cd(
                       to_tsvector('english', coalesce(title,'') || ' ' || coalesce(abstract,'')),
                       websearch_to_tsquery('english', $1)
                   ) AS rank
            FROM research_papers
            WHERE to_tsvector('english', coalesce(title,'') || ' ' || coalesce(abstract,'')) @@
                  websearch_to_tsquery('english', $1)
              AND ($2::text IS NULL OR relevance_status = $2)
            ORDER BY rank DESC NULLS LAST
            LIMIT $3
            """,
            query,
            filters.get("relevance_status"),
            top_k,
        )
        return [
            RetrievedDoc(
                id=f"sql:paper:{row['paper_id']}",
                collection=RetrievalCollection.PAPERS,
                source_table="research_papers",
                source_id=str(row["paper_id"]),
                title=str(row.get("title") or row["paper_id"]),
                text=f"{row.get('title') or ''}\n\n{row.get('abstract') or ''}",
                score=float(row.get("rank") or 0.0),
                sql_rank=rank,
                payload=dict(row),
            )
            for rank, row in enumerate(rows, start=1)
        ]

    async def _sql_search_resources(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any],
    ) -> list[RetrievedDoc]:
        rows = await self.db.fetch(
            """
            SELECT resource_id, title, description, source_platform, resource_type,
                   COALESCE(access_url, source_url, download_url) AS url,
                   ts_rank_cd(
                       to_tsvector('english', coalesce(title,'') || ' ' || coalesce(description,'')),
                       websearch_to_tsquery('english', $1)
                   ) AS rank
            FROM resources
            WHERE to_tsvector('english', coalesce(title,'') || ' ' || coalesce(description,'')) @@
                  websearch_to_tsquery('english', $1)
              AND ($2::text IS NULL OR source_platform = $2)
            ORDER BY rank DESC NULLS LAST
            LIMIT $3
            """,
            query,
            filters.get("source_platform"),
            top_k,
        )
        return [
            RetrievedDoc(
                id=f"sql:resource:{row['resource_id']}",
                collection=RetrievalCollection.RESOURCES,
                source_table="resources",
                source_id=str(row["resource_id"]),
                title=str(row.get("title") or row["resource_id"]),
                text=f"{row.get('title') or ''}\n\n{row.get('description') or ''}",
                score=float(row.get("rank") or 0.0),
                sql_rank=rank,
                payload=dict(row),
            )
            for rank, row in enumerate(rows, start=1)
        ]

    async def _sql_search_programs(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any],
    ) -> list[RetrievedDoc]:
        rows = await self.db.fetch(
            """
            SELECT p.program_id, p.program_title, p.cip_code, p.allied_health_category,
                   i.institution_name, i.state,
                   ts_rank_cd(
                       to_tsvector(
                           'english',
                           coalesce(p.program_title,'') || ' ' || coalesce(i.institution_name,'')
                       ),
                       websearch_to_tsquery('english', $1)
                   ) AS rank
            FROM programs p
            LEFT JOIN institutions i ON p.unitid = i.unitid
            WHERE to_tsvector(
                      'english',
                      coalesce(p.program_title,'') || ' ' || coalesce(i.institution_name,'')
                  ) @@ websearch_to_tsquery('english', $1)
              AND p.allied_health_category IN ('Core Allied Health', 'Allied Health Adjacent')
              AND ($2::text IS NULL OR i.state = $2)
            ORDER BY rank DESC NULLS LAST
            LIMIT $3
            """,
            query,
            filters.get("state"),
            top_k,
        )
        return [
            RetrievedDoc(
                id=f"sql:program:{row['program_id']}",
                collection=RetrievalCollection.PROGRAMS,
                source_table="programs",
                source_id=str(row["program_id"]),
                title=str(row.get("program_title") or row["program_id"]),
                text=f"{row.get('program_title') or ''} at {row.get('institution_name') or ''}",
                score=float(row.get("rank") or 0.0),
                sql_rank=rank,
                payload=dict(row),
            )
            for rank, row in enumerate(rows, start=1)
        ]

    async def _sql_search_communities(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any],
    ) -> list[RetrievedDoc]:
        rows = await self.db.fetch(
            """
            SELECT county_fips, county_name, state, is_priority_county,
                   poverty_percentage, uninsured_percentage,
                   ts_rank_cd(
                       to_tsvector('english', coalesce(county_name,'') || ' ' || coalesce(state,'')),
                       websearch_to_tsquery('english', $1)
                   ) AS rank
            FROM county_profiles
            WHERE (
                    to_tsvector('english', coalesce(county_name,'') || ' ' || coalesce(state,''))
                        @@ websearch_to_tsquery('english', $1)
                    OR ($1 ILIKE '%%rural%%' AND is_priority_county = TRUE)
                    OR ($1 ILIKE '%%georgia%%' AND state = 'GA')
                    OR ($1 ILIKE '%%ga %%' AND state = 'GA')
                  )
              AND ($2::text IS NULL OR state = $2)
            ORDER BY is_priority_county DESC NULLS LAST, rank DESC NULLS LAST
            LIMIT $3
            """,
            query,
            filters.get("state", "GA"),
            top_k,
        )
        return [
            RetrievedDoc(
                id=f"sql:county:{row['county_fips']}",
                collection=RetrievalCollection.COMMUNITIES,
                source_table="county_profiles",
                source_id=str(row["county_fips"]),
                title=f"{row.get('county_name')}, {row.get('state')}",
                text=(
                    f"{row.get('county_name')}, {row.get('state')} "
                    f"poverty={row.get('poverty_percentage')} "
                    f"uninsured={row.get('uninsured_percentage')}"
                ),
                score=float(row.get("rank") or 0.1),
                sql_rank=rank,
                payload=dict(row),
            )
            for rank, row in enumerate(rows, start=1)
        ]

    async def _sql_search_simulation_cases(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any],
    ) -> list[RetrievedDoc]:
        rows = await self.db.fetch(
            """
            SELECT simulation_case_id, title, abstract_or_summary, learning_objectives,
                   case_type, difficulty_level, source_url,
                   ts_rank_cd(
                       to_tsvector(
                           'english',
                           coalesce(title,'') || ' ' || coalesce(abstract_or_summary,'')
                       ),
                       websearch_to_tsquery('english', $1)
                   ) AS rank
            FROM simulation_cases
            WHERE to_tsvector(
                      'english',
                      coalesce(title,'') || ' ' || coalesce(abstract_or_summary,'')
                  ) @@ websearch_to_tsquery('english', $1)
            ORDER BY rank DESC NULLS LAST
            LIMIT $2
            """,
            query,
            top_k,
        )
        return [
            RetrievedDoc(
                id=f"sql:simulation:{row['simulation_case_id']}",
                collection=RetrievalCollection.SIMULATION_CASES,
                source_table="simulation_cases",
                source_id=str(row["simulation_case_id"]),
                title=str(row.get("title") or row["simulation_case_id"]),
                text=f"{row.get('title') or ''}\n\n{row.get('abstract_or_summary') or ''}",
                score=float(row.get("rank") or 0.0),
                sql_rank=rank,
                payload=dict(row),
            )
            for rank, row in enumerate(rows, start=1)
        ]


def reciprocal_rank_fusion(
    vector_results: list[RetrievedDoc],
    sql_results: list[RetrievedDoc],
    top_k: int,
    alpha: float = 0.6,
    k: int = 60,
) -> list[RetrievedDoc]:
    by_source: dict[tuple[str, str], RetrievedDoc] = {}
    scores: defaultdict[tuple[str, str], float] = defaultdict(float)

    for rank, doc in enumerate(vector_results, start=1):
        key = (doc.source_table, doc.source_id)
        by_source[key] = doc
        scores[key] += alpha * (1.0 / (k + (doc.vector_rank or rank)))

    for rank, doc in enumerate(sql_results, start=1):
        key = (doc.source_table, doc.source_id)
        if key not in by_source:
            by_source[key] = doc
        else:
            existing = by_source[key]
            existing.sql_rank = doc.sql_rank or rank
            existing.payload = {**doc.payload, **existing.payload}
        scores[key] += (1.0 - alpha) * (1.0 / (k + (doc.sql_rank or rank)))

    fused = []
    for key, score in scores.items():
        doc = by_source[key]
        fused.append(doc.model_copy(update={"score": score}))
    return sorted(fused, key=lambda doc: doc.score, reverse=True)[:top_k]


def _qdrant_filter(filters: dict[str, Any]) -> models.Filter | None:
    conditions: list[models.FieldCondition] = []
    for key, value in filters.items():
        if value is None or key in {"relevance_status", "source_platform"}:
            continue
        if isinstance(value, (bool, str, int)):
            conditions.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
    return models.Filter(must=conditions) if conditions else None
