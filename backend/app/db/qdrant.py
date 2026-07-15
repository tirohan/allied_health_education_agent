from qdrant_client import AsyncQdrantClient


class QdrantStore:
    """Owns the async Qdrant client lifecycle."""

    def __init__(self, url: str, api_key: str | None = None) -> None:
        self.client = AsyncQdrantClient(url=url, api_key=api_key, timeout=10)

    async def close(self) -> None:
        await self.client.close()
