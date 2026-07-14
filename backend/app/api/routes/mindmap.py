import json
import time
from collections.abc import AsyncIterator

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

from backend.app.agents.graph import compiled_graph, run_mindmap_pipeline
from backend.app.agents.state import MindMapState
from backend.app.schemas.api import MindMapRequest, MindMapResponse

router = APIRouter(prefix="/api/v1", tags=["mindmap"])


@router.post("/mindmap")
async def generate_mindmap(request: Request, body: MindMapRequest) -> MindMapResponse:
    start = time.perf_counter()
    services = request.app.state.services
    initial_state = _initial_state(body)
    final_state = await run_mindmap_pipeline(initial_state, services)
    graph = final_state["mindmap_graph"]
    if graph is None:
        raise RuntimeError("Mind map pipeline completed without a graph")
    return MindMapResponse(
        graph=graph,
        citations=final_state.get("citations", []),
        agent_trace=final_state.get("agent_trace", []),
        total_sources_queried=len(final_state.get("retrieved_docs", [])),
        processing_time_ms=int((time.perf_counter() - start) * 1000),
        cached=False,
    )


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
