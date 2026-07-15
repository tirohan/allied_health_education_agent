import asyncio
from dataclasses import dataclass
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import models

from backend.app.agents.state import RetrievalCollection
from backend.app.db.postgres import Postgres
from backend.app.db.qdrant import QdrantStore
from backend.app.db.redis import RedisCache
from backend.app.retrieval.embedder import Embedder


PAYLOAD_INDEXES: dict[RetrievalCollection, dict[str, models.PayloadSchemaType]] = {
    RetrievalCollection.PAPERS: {
        "source_database": models.PayloadSchemaType.KEYWORD,
        "publication_year": models.PayloadSchemaType.INTEGER,
        "relevance_status": models.PayloadSchemaType.KEYWORD,
        "ai_use_permitted": models.PayloadSchemaType.BOOL,
    },
    RetrievalCollection.RESOURCES: {
        "source_platform": models.PayloadSchemaType.KEYWORD,
        "resource_type": models.PayloadSchemaType.KEYWORD,
        "simulation_use_flag": models.PayloadSchemaType.BOOL,
        "ai_use_permitted": models.PayloadSchemaType.BOOL,
    },
    RetrievalCollection.PROGRAMS: {
        "cip_code": models.PayloadSchemaType.KEYWORD,
        "allied_health_category": models.PayloadSchemaType.KEYWORD,
        "state": models.PayloadSchemaType.KEYWORD,
    },
    RetrievalCollection.COMMUNITIES: {
        "county_fips": models.PayloadSchemaType.KEYWORD,
        "state": models.PayloadSchemaType.KEYWORD,
        "is_priority_county": models.PayloadSchemaType.BOOL,
    },
    RetrievalCollection.SIMULATION_CASES: {
        "case_type": models.PayloadSchemaType.KEYWORD,
        "difficulty_level": models.PayloadSchemaType.KEYWORD,
    },
}


@dataclass(frozen=True)
class IndexResult:
    collection: RetrievalCollection
    indexed: int
    skipped: int = 0


class QdrantIndexer:
    def __init__(
        self,
        db: Postgres,
        qdrant: QdrantStore,
        redis: RedisCache,
        embedder: Embedder,
        vector_size: int,
    ) -> None:
        self.db = db
        self.qdrant = qdrant
        self.redis = redis
        self.embedder = embedder
        self.vector_size = vector_size

    async def ensure_collections(self) -> None:
        for collection in RetrievalCollection:
            await self._ensure_collection(collection)

    async def index_collection(
        self,
        collection: RetrievalCollection,
        batch_size: int = 500,
        limit: int | None = None,
        mode: str = "incremental",
    ) -> IndexResult:
        await self._ensure_collection(collection)
        fetchers = {
            RetrievalCollection.PAPERS: self._fetch_papers,
            RetrievalCollection.RESOURCES: self._fetch_resources,
            RetrievalCollection.PROGRAMS: self._fetch_programs,
            RetrievalCollection.COMMUNITIES: self._fetch_communities,
            RetrievalCollection.SIMULATION_CASES: self._fetch_simulation_cases,
        }
        rows = await fetchers[collection](limit)

        checkpoint_key = f"index:{collection.value}:last_offset"
        start_offset = 0
        if mode == "incremental":
            checkpoint = await self.redis.get_json(checkpoint_key)
            try:
                start_offset = int(checkpoint) if checkpoint is not None else 0
            except ValueError:
                start_offset = 0
        rows = rows[start_offset:]

        indexed = 0
        skipped = 0
        for local_offset in range(0, len(rows), batch_size):
            batch = rows[local_offset : local_offset + batch_size]
            points = await self._rows_to_points(collection, batch)
            if points:
                await self.qdrant.client.upsert(collection_name=collection.value, points=points)
                indexed += len(points)
            skipped += len(batch) - len(points)
            await self.redis.set_checkpoint(
                checkpoint_key,
                start_offset + local_offset + len(batch),
            )
            # Pace large OpenAI embedding jobs to stay under TPM limits.
            if collection == RetrievalCollection.PAPERS and local_offset + batch_size < len(rows):
                await asyncio.sleep(0.75)
        return IndexResult(collection=collection, indexed=indexed, skipped=skipped)

    async def _ensure_collection(self, collection: RetrievalCollection) -> None:
        exists = await self.qdrant.client.collection_exists(collection.value)
        if not exists:
            await self.qdrant.client.create_collection(
                collection_name=collection.value,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
        for field_name, field_schema in PAYLOAD_INDEXES.get(collection, {}).items():
            await self.qdrant.client.create_payload_index(
                collection_name=collection.value,
                field_name=field_name,
                field_schema=field_schema,
            )

    async def _fetch_papers(self, limit: int | None) -> list[dict[str, Any]]:
        sql = """
        SELECT paper_id, title, abstract, publication_year, relevance_status,
               evidence_level, source_database, doi,
               COALESCE(
                   ai_use_permitted,
                   CASE
                       WHEN source_database ILIKE '%%openalex%%' THEN TRUE
                       WHEN source_database ILIKE '%%pubmed%%' THEN TRUE
                       WHEN source_database ILIKE '%%mededportal%%' THEN TRUE
                       ELSE NULL
                   END
               ) AS ai_use_permitted
        FROM research_papers
        WHERE coalesce(title, '') <> ''
        ORDER BY paper_id
        """
        if limit:
            sql += f" LIMIT {int(limit)}"
        return [dict(row) for row in await self.db.fetch(sql)]

    async def _fetch_resources(self, limit: int | None) -> list[dict[str, Any]]:
        sql = """
        SELECT resource_id, title, description, source_platform, resource_type,
               simulation_use_flag, license_category,
               COALESCE(access_url, source_url, download_url) AS url,
               COALESCE(
                   ai_use_permitted,
                   CASE
                       WHEN license_category ILIKE 'Open%%' THEN TRUE
                       WHEN license_category ILIKE '%%Public Domain%%' THEN TRUE
                       WHEN source_platform IN ('CDC', 'SAMHSA', 'AHRQ') THEN TRUE
                       ELSE NULL
                   END
               ) AS ai_use_permitted
        FROM resources
        WHERE coalesce(title, '') <> ''
        ORDER BY resource_id
        """
        if limit:
            sql += f" LIMIT {int(limit)}"
        return [dict(row) for row in await self.db.fetch(sql)]

    async def _fetch_programs(self, limit: int | None) -> list[dict[str, Any]]:
        sql = """
        SELECT p.program_id, p.program_title, p.cip_code, p.allied_health_category,
               p.unitid, i.institution_name, i.state,
               COALESCE(p.program_title, '') || ' ' || COALESCE(i.institution_name, '')
                   || ' ' || COALESCE(p.cip_code, '') AS text
        FROM programs p
        LEFT JOIN institutions i ON p.unitid = i.unitid
        WHERE coalesce(p.program_title, '') <> ''
          AND p.allied_health_category IN ('Core Allied Health', 'Allied Health Adjacent')
        ORDER BY p.program_id
        """
        if limit:
            sql += f" LIMIT {int(limit)}"
        return [dict(row) for row in await self.db.fetch(sql)]

    async def _fetch_communities(self, limit: int | None) -> list[dict[str, Any]]:
        sql = """
        SELECT c.county_fips, c.county_name, c.state, c.is_priority_county,
               c.poverty_percentage, c.uninsured_percentage,
               COALESCE(c.county_name, '') || ' County ' || COALESCE(c.state, '')
                   || ' poverty ' || COALESCE(c.poverty_percentage::text, '')
                   || ' uninsured ' || COALESCE(c.uninsured_percentage::text, '') AS text
        FROM county_profiles c
        WHERE c.state = 'GA'
        ORDER BY c.county_fips
        """
        if limit:
            sql += f" LIMIT {int(limit)}"
        return [dict(row) for row in await self.db.fetch(sql)]

    async def _fetch_simulation_cases(self, limit: int | None) -> list[dict[str, Any]]:
        sql = """
        SELECT simulation_case_id, title, abstract_or_summary, learning_objectives,
               case_type, difficulty_level, source_url, topic_tags
        FROM simulation_cases
        WHERE coalesce(title, '') <> ''
        ORDER BY simulation_case_id
        """
        if limit:
            sql += f" LIMIT {int(limit)}"
        return [dict(row) for row in await self.db.fetch(sql)]

    async def _rows_to_points(
        self,
        collection: RetrievalCollection,
        rows: list[dict[str, Any]],
    ) -> list[models.PointStruct]:
        candidates = [
            (row, text)
            for row in rows
            if (text := self._document_text(collection, row)).strip()
        ]
        if not candidates:
            return []
        vectors = await self.embedder.embed_many([text for _, text in candidates])
        points: list[models.PointStruct] = []
        for (row, text), vector in zip(candidates, vectors, strict=True):
            point_id = self._point_id(collection, row)
            payload = self._payload(collection, row, text)
            points.append(models.PointStruct(id=point_id, vector=vector, payload=payload))
        return points

    def _document_text(self, collection: RetrievalCollection, row: dict[str, Any]) -> str:
        if collection == RetrievalCollection.PAPERS:
            return f"{row.get('title') or ''}\n\n{row.get('abstract') or ''}"
        if collection == RetrievalCollection.RESOURCES:
            return f"{row.get('title') or ''}\n\n{row.get('description') or ''}"
        if collection == RetrievalCollection.PROGRAMS:
            return str(row.get("text") or row.get("program_title") or "")
        if collection == RetrievalCollection.COMMUNITIES:
            return str(row.get("text") or row.get("county_name") or "")
        if collection == RetrievalCollection.SIMULATION_CASES:
            return (
                f"{row.get('title') or ''}\n\n"
                f"{row.get('abstract_or_summary') or ''}\n\n"
                f"{row.get('learning_objectives') or ''}"
            )
        return str(row)

    def _point_id(self, collection: RetrievalCollection, row: dict[str, Any]) -> str:
        key_map = {
            RetrievalCollection.PAPERS: ("paper", "paper_id"),
            RetrievalCollection.RESOURCES: ("resource", "resource_id"),
            RetrievalCollection.PROGRAMS: ("program", "program_id"),
            RetrievalCollection.COMMUNITIES: ("county", "county_fips"),
            RetrievalCollection.SIMULATION_CASES: ("simulation", "simulation_case_id"),
        }
        prefix, field = key_map[collection]
        return str(uuid5(NAMESPACE_URL, f"{prefix}:{row[field]}"))

    def _payload(
        self,
        collection: RetrievalCollection,
        row: dict[str, Any],
        text: str,
    ) -> dict[str, Any]:
        if collection == RetrievalCollection.PAPERS:
            return {
                "source_table": "research_papers",
                "source_id": row["paper_id"],
                "title": row.get("title"),
                "text": text[:4_000],
                "publication_year": row.get("publication_year"),
                "relevance_status": row.get("relevance_status"),
                "evidence_level": row.get("evidence_level"),
                "ai_use_permitted": row.get("ai_use_permitted"),
                "source_database": row.get("source_database"),
                "doi": row.get("doi"),
            }
        if collection == RetrievalCollection.RESOURCES:
            return {
                "source_table": "resources",
                "source_id": row["resource_id"],
                "title": row.get("title"),
                "text": text[:4_000],
                "source_platform": row.get("source_platform"),
                "resource_type": row.get("resource_type"),
                "simulation_use_flag": row.get("simulation_use_flag"),
                "license_category": row.get("license_category"),
                "ai_use_permitted": row.get("ai_use_permitted"),
                "url": row.get("url"),
            }
        if collection == RetrievalCollection.PROGRAMS:
            return {
                "source_table": "programs",
                "source_id": row["program_id"],
                "title": row.get("program_title"),
                "text": text[:4_000],
                "cip_code": row.get("cip_code"),
                "allied_health_category": row.get("allied_health_category"),
                "state": row.get("state"),
                "institution_name": row.get("institution_name"),
                "unitid": row.get("unitid"),
            }
        if collection == RetrievalCollection.COMMUNITIES:
            return {
                "source_table": "county_profiles",
                "source_id": row["county_fips"],
                "title": f"{row.get('county_name')}, {row.get('state')}",
                "text": text[:4_000],
                "county_fips": row.get("county_fips"),
                "state": row.get("state"),
                "is_priority_county": row.get("is_priority_county"),
                "poverty_percentage": float(row["poverty_percentage"])
                if row.get("poverty_percentage") is not None
                else None,
                "uninsured_percentage": float(row["uninsured_percentage"])
                if row.get("uninsured_percentage") is not None
                else None,
            }
        if collection == RetrievalCollection.SIMULATION_CASES:
            return {
                "source_table": "simulation_cases",
                "source_id": row["simulation_case_id"],
                "title": row.get("title"),
                "text": text[:4_000],
                "case_type": row.get("case_type"),
                "difficulty_level": row.get("difficulty_level"),
                "url": row.get("source_url"),
                "topic_tags": row.get("topic_tags"),
            }
        return row
