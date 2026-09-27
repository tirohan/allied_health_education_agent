import re

from backend.app.agents.state import AgentStep, MindMapState, RetrievalCollection

# The county profile and county shortage summary layers are loaded for Georgia only.
COUNTY_LAYER_STATE = "GA"
US_STATES: dict[str, str] = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA",
    "colorado": "CO", "connecticut": "CT", "delaware": "DE", "district of columbia": "DC",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID", "illinois": "IL",
    "indiana": "IN", "iowa": "IA", "kansas": "KS", "kentucky": "KY", "louisiana": "LA",
    "maine": "ME", "maryland": "MD", "massachusetts": "MA", "michigan": "MI",
    "minnesota": "MN", "mississippi": "MS", "missouri": "MO", "montana": "MT",
    "nebraska": "NE", "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ",
    "new mexico": "NM", "new york": "NY", "north carolina": "NC", "north dakota": "ND",
    "ohio": "OH", "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA",
    "rhode island": "RI", "south carolina": "SC", "south dakota": "SD", "tennessee": "TN",
    "texas": "TX", "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY", "puerto rico": "PR",
}
_STATE_RE = re.compile(
    r"\b(" + "|".join(sorted((re.escape(n) for n in US_STATES), key=len, reverse=True)) + r")\b"
)


# The county layer and program search only answer questions that ask about
# place/need or about programs; otherwise they add unrelated nodes (e.g. Georgia
# counties on a sonography query), as the 2026-09-27 live audit showed.
_PLACE_TOKENS = (
    "georgia", "county", "counties", "rural", "hpsa", "shortage", "underserved",
    "community", "communities", "region", "local",
)
_PROGRAM_TOKENS = (
    "program", "degree", "school", "college", "universit", "institution", "workforce",
    "pipeline", "enroll", "accredit", "offered", "available", "graduate",
)


def _mentions(lowered: str, tokens: tuple[str, ...]) -> bool:
    return any(re.search(rf"\b{token}", lowered) for token in tokens)


def states_named(query: str) -> list[str]:
    """Postal codes of U.S. states named in the query, in order of appearance."""
    lowered = query.lower()
    # Longest names first, non-overlapping: "west virginia" never also yields "virginia".
    codes = [US_STATES[m.group(1)] for m in _STATE_RE.finditer(lowered)]
    return list(dict.fromkeys(codes))


def county_scope_note(state_codes: list[str]) -> str:
    names = ", ".join(state_codes)
    return (
        f"The county profile and county shortage summary layer covers Georgia only. "
        f"No county-level community or shortage summary data is available for {names}; "
        f"program and institution results are not limited to Georgia."
    )


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
    names_state = bool(states_named(query)) or " ga" in f" {lowered}"
    gated_out = []
    if not (_mentions(lowered, _PLACE_TOKENS) or names_state):
        gated_out.append(RetrievalCollection.COMMUNITIES)
    if not (_mentions(lowered, _PROGRAM_TOKENS) or _mentions(lowered, _PLACE_TOKENS) or names_state):
        gated_out.append(RetrievalCollection.PROGRAMS)
    inferred = [c for c in inferred if c not in gated_out] or [
        RetrievalCollection.PAPERS,
        RetrievalCollection.RESOURCES,
    ]
    filters = dict(state.get("filters") or {})
    if "georgia" in lowered or " ga" in f" {lowered}":
        filters.setdefault("state", "GA")
    scope_notes = list(state.get("scope_notes") or [])
    other_states = [code for code in states_named(query) if code != COUNTY_LAYER_STATE]
    if other_states:
        scope_notes.append(county_scope_note(other_states))
        if filters.get("state") != COUNTY_LAYER_STATE:
            # Georgia county rows would otherwise answer an out-of-state question
            # (e.g. "rural" matches GA priority counties). Drop the GA-only layer
            # and keep the nationwide program/institution search.
            inferred = [c for c in inferred if c != RetrievalCollection.COMMUNITIES]
            if RetrievalCollection.PROGRAMS not in inferred:
                inferred.append(RetrievalCollection.PROGRAMS)
    return {
        **state,
        "refined_query": query,
        "collections": inferred,
        "filters": filters,
        "scope_notes": scope_notes,
        "agent_trace": [
            *state.get("agent_trace", []),
            AgentStep(
                agent="orchestrator",
                message="Classified query for hybrid retrieval and mind map extraction.",
                metadata={
                    "collections": [collection.value for collection in inferred],
                    "filters": filters,
                    "scope_notes": scope_notes,
                    "gated_out": [collection.value for collection in gated_out],
                },
            ),
        ],
    }
