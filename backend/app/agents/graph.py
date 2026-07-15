from collections.abc import Awaitable, Callable
from typing import Any

import structlog
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from backend.app.agents.extraction import extraction_node
from backend.app.agents.mindmap import mindmap_node
from backend.app.agents.orchestrator import orchestrator_node
from backend.app.agents.research import research_node
from backend.app.agents.state import MindMapState
from backend.app.agents.verification import verification_node

logger = structlog.get_logger(__name__)


def _guard_state_only(
    name: str, fn: Callable[[MindMapState], Awaitable[MindMapState]]
) -> Callable[[MindMapState], Awaitable[MindMapState]]:
    async def wrapper(state: MindMapState) -> MindMapState:
        if state.get("error"):
            return state
        try:
            return await fn(state)
        except Exception as exc:  # noqa: BLE001 - never let one node crash the whole run
            logger.error("agent_node_failed", agent=name, error=str(exc))
            return {**state, "error": f"{name}: {exc}"}

    return wrapper


def _guard_with_config(
    name: str,
    fn: Callable[[MindMapState, RunnableConfig], Awaitable[MindMapState]],
) -> Callable[[MindMapState, RunnableConfig], Awaitable[MindMapState]]:
    async def wrapper(state: MindMapState, config: RunnableConfig) -> MindMapState:
        if state.get("error"):
            return state
        try:
            return await fn(state, config)
        except Exception as exc:  # noqa: BLE001 - never let one node crash the whole run
            logger.error("agent_node_failed", agent=name, error=str(exc))
            return {**state, "error": f"{name}: {exc}"}

    return wrapper


def build_graph():
    graph = StateGraph(MindMapState)
    graph.add_node("orchestrator", _guard_state_only("orchestrator", orchestrator_node))
    graph.add_node("research", _guard_with_config("research", research_node))
    graph.add_node("extraction", _guard_with_config("extraction", extraction_node))
    graph.add_node("verification", _guard_with_config("verification", verification_node))
    graph.add_node("mindmap", _guard_state_only("mindmap", mindmap_node))

    graph.set_entry_point("orchestrator")
    graph.add_edge("orchestrator", "research")
    graph.add_edge("research", "extraction")
    graph.add_edge("extraction", "verification")
    graph.add_edge("verification", "mindmap")
    graph.add_edge("mindmap", END)
    return graph.compile()


compiled_graph = build_graph()


async def run_mindmap_pipeline(initial_state: MindMapState, services: Any) -> MindMapState:
    return await compiled_graph.ainvoke(
        initial_state,
        config={"configurable": {"services": services}},
    )
