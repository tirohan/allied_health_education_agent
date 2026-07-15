import hashlib
import json
import time
from collections.abc import AsyncIterator

import structlog
from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from backend.app.agents.graph import compiled_graph, run_mindmap_pipeline
from backend.app.agents.state import MindMapState
from backend.app.schemas.api import MindMapRequest, MindMapResponse

router = APIRouter(prefix="/api/v1", tags=["mindmap"])
logger = structlog.get_logger(__name__)


def _cache_key(body: MindMapRequest) -> str:
    payload = json.dumps(body.model_dump(mode="json"), sort_keys=True)
    return "mindmap:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


@router.post("/mindmap")
async def generate_mindmap(request: Request, body: MindMapRequest) -> MindMapResponse:
    start = time.perf_counter()
    services = request.app.state.services
    cache_key = _cache_key(body)

    try:
        cached_json = await services.redis.get_json(cache_key)
    except Exception as exc:  # noqa: BLE001 - a cache outage must not break the request
        cached_json = None
        logger.warning("mindmap_cache_read_failed", error=str(exc))

    if cached_json is not None:
        try:
            cached_response = MindMapResponse.model_validate_json(cached_json)
            cached_response.cached = True
            return cached_response
        except Exception as exc:  # noqa: BLE001 - fall through to a fresh computation
            logger.warning("mindmap_cache_deserialize_failed", error=str(exc))

    initial_state = _initial_state(body)
    final_state = await run_mindmap_pipeline(initial_state, services)
    if final_state.get("error"):
        logger.error("mindmap_pipeline_failed", error=final_state["error"])
        raise HTTPException(
            status_code=502, detail=f"Mind map generation failed: {final_state['error']}"
        )
    graph = final_state["mindmap_graph"]
    if graph is None:
        raise HTTPException(
            status_code=500, detail="Mind map pipeline completed without a graph"
        )
    response = MindMapResponse(
        graph=graph,
        citations=final_state.get("citations", []),
        agent_trace=final_state.get("agent_trace", []),
        total_sources_queried=len(final_state.get("retrieved_docs", [])),
        processing_time_ms=int((time.perf_counter() - start) * 1000),
        cached=False,
    )

    try:
        await services.redis.set_json(
            cache_key, response.model_dump_json(), services.settings.cache_ttl_seconds
        )
    except Exception as exc:  # noqa: BLE001 - a cache outage must not break the request
        logger.warning("mindmap_cache_write_failed", error=str(exc))

    return response


@router.get("/mindmap/stream")
async def stream_mindmap(
    request: Request,
    query: str,
    max_nodes: int = 50,
) -> EventSourceResponse:
    body = MindMapRequest(query=query, max_nodes=max_nodes)

    async def events() -> AsyncIterator[dict[str, str]]:
        state = _initial_state(body)
        async for event in compiled_graph.astream(
            state,
            config={"configurable": {"services": request.app.state.services}},
            stream_mode="updates",
        ):
            yield {"event": "agent_update", "data": json.dumps(_jsonable(event))}
        yield {"event": "complete", "data": "{}"}

    return EventSourceResponse(events())


def _initial_state(body: MindMapRequest) -> MindMapState:
    return {
        "query": body.query,
        "refined_query": "",
        "collections": body.collections,
        "filters": body.filters,
        "max_nodes": body.max_nodes,
        "min_confidence": body.min_confidence,
        "extraction_mode": body.extraction_mode,
        "retrieval_mode": body.retrieval_mode,
        "retrieved_docs": [],
        "extracted_entities": [],
        "extracted_relations": [],
        "verification_results": [],
        "mindmap_graph": None,
        "citations": [],
        "confidence_scores": {},
        "agent_trace": [],
        "error": None,
    }


def _jsonable(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value
