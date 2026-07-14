from typing import Any

from langgraph.graph import END, StateGraph

from backend.app.agents.extraction import extraction_node
from backend.app.agents.mindmap import mindmap_node
from backend.app.agents.orchestrator import orchestrator_node
from backend.app.agents.research import research_node
from backend.app.agents.state import MindMapState
from backend.app.agents.verification import verification_node


def build_graph():
    graph = StateGraph(MindMapState)
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("research", research_node)
    graph.add_node("extraction", extraction_node)
    graph.add_node("verification", verification_node)
    graph.add_node("mindmap", mindmap_node)

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
