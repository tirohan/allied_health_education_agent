import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader

from backend.app.core.config import Settings, get_settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(
    provided_key: str | None = Depends(_api_key_header),
    settings: Settings = Depends(get_settings),
) -> None:
    """Optional API-key auth.

    If `settings.api_key` is unset, this is a no-op (auth stays fully open,
    preserving today's local-dev behavior). If it is set, the caller must
    send a matching `X-API-Key` header.
    """

    if settings.api_key is None:
        return

    expected_key = settings.api_key.get_secret_value()
    if provided_key is None or not secrets.compare_digest(provided_key, expected_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
