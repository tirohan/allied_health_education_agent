from langchain_core.runnables import RunnableConfig

from backend.app.agents.state import AgentStep, MindMapState, RetrievalCollection
from backend.app.retrieval.hybrid import HybridSearch


async def research_node(state: MindMapState, config: RunnableConfig) -> MindMapState:
    services = config["configurable"]["services"]
    search = HybridSearch(services.postgres, services.qdrant, services.embedder)
    retrieval_mode = str(state.get("retrieval_mode") or "hybrid")
    # Retrieval depth is deliberately decoupled from max_nodes: max_nodes caps the
    # *rendered* graph size (applied later during pruning in mindmap.py), but
    # retrieval needs a wider pool of candidate documents to extract from and
    # verify, regardless of how small the final map should be.
    retrieval_top_k = max(int(state.get("max_nodes") or 50) * 3, 30)
    retrieval_top_k = min(retrieval_top_k, 150)
    docs = await search.search(
        query=state.get("refined_query") or state["query"],
        collections=[RetrievalCollection(collection) for collection in state.get("collections", [])],
        top_k=retrieval_top_k,
        filters=state.get("filters", {}),
        mode=retrieval_mode,
    )
    return {
        **state,
        "retrieved_docs": docs,
        "agent_trace": [
            *state.get("agent_trace", []),
            AgentStep(
                agent="research",
                message=f"Retrieved {len(docs)} documents with {retrieval_mode} search.",
                metadata={
                    "top_score": docs[0].score if docs else 0.0,
                    "retrieval_mode": retrieval_mode,
                },
            ),
        ],
    }
