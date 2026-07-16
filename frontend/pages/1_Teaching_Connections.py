import streamlit as st

from frontend.api_client import ApiError, build_mindmap_with_progress, post, safe_call
from frontend.components.educator_panel import render_educator_node_card, render_plain_evidence_list
from frontend.components.graph_viz import render_mindmap, render_trust_legend
from frontend.components.guided_starters import render_guided_starters
from frontend.components.role_selector import render_role_selector
from frontend.components.teaching_list import render_teaching_list_sidebar

st.set_page_config(page_title="Teaching Connections", layout="wide")
st.title("Teaching Connections")
st.caption(
    "Turn a teaching goal into a connected plan. "
    "Ask what you can teach, why you can trust it, and what to do next."
)

with st.sidebar:
    st.header("Your role")
    role = render_role_selector()
    advanced_mode = st.toggle(
        "Advanced / Research mode",
        value=st.session_state.get("advanced_mode", False),
    )
    st.session_state["advanced_mode"] = advanced_mode
    st.divider()
    st.header("Source focus")
    collection_labels = {
        "papers": "Research articles",
        "resources": "Teaching resources",
        "programs": "Degree programs",
        "communities": "Georgia counties",
        "simulation_cases": "Simulation cases",
    }
    default_collections = st.session_state.get(
        "planning_collections",
        ["papers", "resources", "programs", "communities", "simulation_cases"],
    )
    collections = st.multiselect(
        "Include",
        options=list(collection_labels.keys()),
        default=default_collections,
        format_func=lambda key: collection_labels[key],
    )
    max_nodes = st.slider("How many items to show", min_value=8, max_value=60, value=30)

render_teaching_list_sidebar()

starter = render_guided_starters()
if starter:
    st.info(
        f"Workflow ready: {starter['title']}. "
        f"Suggested next step after the map: {starter.get('next_action', 'Open Curriculum Builder')}."
    )
query = st.text_area(
    "Or refine your planning question",
    value=st.session_state.get(
        "planning_query",
        "What interprofessional education resources address opioid education in rural Georgia counties?",
    ),
    height=90,
)

if st.button("Build teaching map", type="primary"):
    try:
        response = build_mindmap_with_progress(
            {
                "query": query,
                "max_nodes": max_nodes,
                "min_confidence": 0.45,
                "collections": collections,
                "filters": st.session_state.get("planning_filters", {"state": "GA"}),
            }
        )
        enrich = safe_call(
            "Explaining what you found...",
            post,
            "/api/v1/educator/enrich",
            {
                "query": query,
                "role": role,
                "graph": response["graph"],
            },
        )
        st.session_state["mindmap_response"] = response
        st.session_state["educator_cards"] = enrich.get("cards", [])
        st.session_state["planning_query"] = query
    except ApiError as exc:
        st.error(f"Couldn't build the teaching map: {exc}")

response = st.session_state.get("mindmap_response")
cards = st.session_state.get("educator_cards", [])

if response:
    st.success(
        f"Found {len(response['graph'].get('nodes', []))} connected teaching items "
        f"from {response.get('total_sources_queried', 0)} sources."
    )
    render_trust_legend(response["graph"])
    left, right = st.columns([0.68, 0.32])
    with left:
        selected = render_mindmap(response["graph"])
        selected_node_id = st.session_state.get("selected_node_id")
        if selected:
            selected_node_id = selected
            st.session_state["selected_node_id"] = selected_node_id
        labels = {
            node["id"]: f"{node.get('entity_type')}: {node.get('label')}"
            for node in response["graph"].get("nodes", [])
        }
        if labels:
            st.caption("Tip: you can also click directly on a bubble in the map above.")
            selected_node_id = st.selectbox(
                "Choose an item to inspect",
                options=list(labels.keys()),
                format_func=lambda key: labels[key],
                index=list(labels.keys()).index(selected_node_id)
                if selected_node_id in labels
                else 0,
            )
            st.session_state["selected_node_id"] = selected_node_id
    with right:
        render_educator_node_card(
            cards,
            st.session_state.get("selected_node_id"),
            advanced_mode=advanced_mode,
        )

    render_plain_evidence_list(cards)

    if advanced_mode:
        with st.expander("Advanced agent steps"):
            st.json(response.get("agent_trace", []))
else:
    st.info(
        "Choose a guided workflow above, then select **Build teaching map**. "
        f"Suggested next step from your starter: "
        f"{(starter or {}).get('next_action', 'Open Curriculum Builder')}."
    )
