from fastapi import APIRouter, Request

from backend.app.retrieval.hybrid import HybridSearch
from backend.app.schemas.api import SearchRequest, SearchResponse

router = APIRouter(prefix="/api/v1", tags=["search"])


@router.post("/search")
async def search(request: Request, body: SearchRequest) -> SearchResponse:
    services = request.app.state.services
    hybrid = HybridSearch(services.postgres, services.qdrant, services.embedder)
    results = await hybrid.search(
        query=body.query,
        collections=body.collections,
        top_k=body.top_k,
        filters=body.filters,
        mode=body.mode,
    )
    return SearchResponse(results=results, total=len(results))
