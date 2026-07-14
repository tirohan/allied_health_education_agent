from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Any

import asyncpg


class Postgres:
    """Small asyncpg wrapper with parameterized query helpers."""

    def __init__(self, database_url: str, min_size: int = 1, max_size: int = 10) -> None:
        self._database_url = database_url
        self._min_size = min_size
        self._max_size = max_size
        self._pool: asyncpg.Pool | None = None

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("PostgreSQL pool has not been started")
        return self._pool

    async def connect(self) -> None:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                dsn=self._database_url,
                min_size=self._min_size,
                max_size=self._max_size,
                command_timeout=60,
            )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def fetch(self, sql: str, *args: object) -> list[Mapping[str, Any]]:
        rows = await self.pool.fetch(sql, *args)
        return [dict(row) for row in rows]

    async def fetchrow(self, sql: str, *args: object) -> Mapping[str, Any] | None:
        row = await self.pool.fetchrow(sql, *args)
        return dict(row) if row is not None else None

    async def execute(self, sql: str, *args: object) -> str:
        return await self.pool.execute(sql, *args)

    async def executemany(self, sql: str, args: Sequence[Sequence[object]]) -> None:
        await self.pool.executemany(sql, args)

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[asyncpg.Connection]:
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                yield connection
