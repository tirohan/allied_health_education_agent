from backend.app.agents.state import AgentStep, MindMapState, RetrievalCollection


async def orchestrator_node(state: MindMapState) -> MindMapState:
    query = state["query"].strip()
    collections = state.get("collections") or [
        RetrievalCollection.PAPERS,
        RetrievalCollection.RESOURCES,
    ]
    lowered = query.lower()
    inferred = list(collections)
    if any(token in lowered for token in ("georgia", "county", "rural", "hpsa", "shortage")):
        for item in (RetrievalCollection.COMMUNITIES, RetrievalCollection.PROGRAMS):
            if item not in inferred:
                inferred.append(item)
    if any(token in lowered for token in ("simulation", "case", "scenario")):
        if RetrievalCollection.SIMULATION_CASES not in inferred:
            inferred.append(RetrievalCollection.SIMULATION_CASES)
    filters = dict(state.get("filters") or {})
    if "georgia" in lowered or " ga" in f" {lowered}":
        filters.setdefault("state", "GA")
    return {
        **state,
        "refined_query": query,
        "collections": inferred,
        "filters": filters,
        "agent_trace": [
            *state.get("agent_trace", []),
            AgentStep(
                agent="orchestrator",
                message="Classified query for hybrid retrieval and mind map extraction.",
                metadata={
                    "collections": [collection.value for collection in inferred],
                    "filters": filters,
                },
            ),
        ],
    }
