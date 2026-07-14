from langchain_core.runnables import RunnableConfig

from backend.app.agents.state import AgentStep, MindMapState, RetrievalCollection
from backend.app.retrieval.hybrid import HybridSearch


async def research_node(state: MindMapState, config: RunnableConfig) -> MindMapState:
    services = config["configurable"]["services"]
    search = HybridSearch(services.postgres, services.qdrant, services.embedder)
    docs = await search.search(
        query=state.get("refined_query") or state["query"],
        collections=[RetrievalCollection(collection) for collection in state.get("collections", [])],
        top_k=state.get("max_nodes", 50),
        filters=state.get("filters", {}),
    )
    return {
        **state,
        "retrieved_docs": docs,
        "agent_trace": [
            *state.get("agent_trace", []),
            AgentStep(
                agent="research",
                message=f"Retrieved {len(docs)} documents with hybrid search.",
                metadata={"top_score": docs[0].score if docs else 0.0},
            ),
        ],
    }
