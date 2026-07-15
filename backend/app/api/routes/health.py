import asyncio

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api/v1", tags=["health"])

_CHECK_TIMEOUT_SECONDS = 3.0


async def _check_postgres(services) -> str:  # type: ignore[no-untyped-def]
    await services.postgres.fetchrow("SELECT 1")
    return "ok"


async def _check_qdrant(services) -> str:  # type: ignore[no-untyped-def]
    await services.qdrant.client.get_collections()
    return "ok"


async def _check_redis(services) -> str:  # type: ignore[no-untyped-def]
    await services.redis.client.ping()
    return "ok"


async def _run_check(coro) -> str:  # type: ignore[no-untyped-def]
    try:
        return await asyncio.wait_for(coro, timeout=_CHECK_TIMEOUT_SECONDS)
    except Exception as exc:  # noqa: BLE001 - surfaced as a dependency status string
        return f"error: {exc}"


@router.get("/health")
async def health(request: Request) -> JSONResponse:
    services = request.app.state.services

    postgres_status, qdrant_status, redis_status = await asyncio.gather(
        _run_check(_check_postgres(services)),
        _run_check(_check_qdrant(services)),
        _run_check(_check_redis(services)),
        return_exceptions=True,
    )

    dependencies = {
        "postgres": postgres_status,
        "qdrant": qdrant_status,
        "redis": redis_status,
    }
    all_ok = all(status == "ok" for status in dependencies.values())

    return JSONResponse(
        status_code=200 if all_ok else 503,
        content={
            "status": "ok" if all_ok else "error",
            "app_env": services.settings.app_env,
            "dependencies": dependencies,
        },
    )
