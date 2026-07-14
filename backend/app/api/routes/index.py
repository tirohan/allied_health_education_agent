from fastapi import APIRouter, Request

from backend.app.retrieval.indexer import QdrantIndexer
from backend.app.schemas.api import IndexRequest, IndexResponse

router = APIRouter(prefix="/api/v1", tags=["index"])


@router.post("/index")
async def index_collection(request: Request, body: IndexRequest) -> IndexResponse:
    services = request.app.state.services
    indexer = QdrantIndexer(
        db=services.postgres,
        qdrant=services.qdrant,
        redis=services.redis,
        embedder=services.embedder,
        vector_size=services.settings.qdrant_vector_size,
    )
    result = await indexer.index_collection(
        collection=body.collection,
        batch_size=body.batch_size,
        limit=body.limit,
    )
    return IndexResponse(
        collection=result.collection,
        status="completed",
        indexed=result.indexed,
        skipped=result.skipped,
    )
