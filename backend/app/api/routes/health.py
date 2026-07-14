from fastapi import APIRouter, Request

from backend.app.schemas.api import HealthResponse

router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health")
async def health(request: Request) -> HealthResponse:
    settings = request.app.state.services.settings
    return HealthResponse(status="ok", app_env=settings.app_env)
