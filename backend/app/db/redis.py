from typing import Any

from redis.asyncio import Redis


class RedisCache:
    """Thin Redis client wrapper for caching and indexing checkpoints."""

    def __init__(self, redis_url: str) -> None:
        self.client: Redis = Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_timeout=10,
            socket_connect_timeout=10,
        )

    async def close(self) -> None:
        await self.client.aclose()

    async def get_json(self, key: str) -> str | None:
        value = await self.client.get(key)
        return str(value) if value is not None else None

    async def set_json(self, key: str, value: str, ttl_seconds: int) -> None:
        await self.client.set(key, value, ex=ttl_seconds if ttl_seconds > 0 else None)

    async def set_checkpoint(self, key: str, value: Any) -> None:
        await self.client.set(key, str(value))
