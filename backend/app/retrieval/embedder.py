import asyncio
import hashlib
import logging

from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, RateLimitError

logger = logging.getLogger(__name__)

_RETRYABLE_ERRORS = (RateLimitError, APIConnectionError, APITimeoutError)


class Embedder:
    """Embedding provider with a deterministic fallback for local tests."""

    def __init__(self, model: str, api_key: str | None, vector_size: int = 1536) -> None:
        self.model = model
        self.vector_size = vector_size
        self._client = AsyncOpenAI(api_key=api_key, timeout=30.0) if api_key else None

    async def embed(self, text: str) -> list[float]:
        vectors = await self.embed_many([text])
        return vectors[0]

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        normalized = [text[:8_000] if text else " " for text in texts]
        if self._client is None:
            return [self._deterministic_vector(text) for text in normalized]

        last_error: Exception | None = None
        for attempt in range(6):
            try:
                response = await self._client.embeddings.create(
                    model=self.model,
                    input=normalized,
                )
                return [item.embedding for item in response.data]
            except _RETRYABLE_ERRORS as exc:
                last_error = exc
                wait_s = min(20.0, 1.5 * (2**attempt))
                logger.warning(
                    "OpenAI embedding call failed (%s, attempt %s); sleeping %.1fs",
                    type(exc).__name__,
                    attempt + 1,
                    wait_s,
                )
                await asyncio.sleep(wait_s)
        assert last_error is not None
        raise last_error

    def _deterministic_vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values: list[float] = []
        while len(values) < self.vector_size:
            for byte in digest:
                values.append((byte / 255.0) * 2.0 - 1.0)
                if len(values) == self.vector_size:
                    break
            digest = hashlib.sha256(digest).digest()
        return values
