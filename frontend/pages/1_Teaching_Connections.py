import streamlit as st

from frontend.api_client import ApiError, build_mindmap_with_progress, post, safe_call
from frontend.components.educator_panel import render_educator_node_card
from frontend.components.graph_viz import ENTITY_LEGEND, render_mindmap, render_trust_legend
from frontend.components.guided_starters import render_guided_starters
from frontend.components.role_selector import render_role_selector
from frontend.components.teaching_list import render_teaching_list_sidebar

CARDS_SHOWN_BY_DEFAULT = 12

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
        st.session_state["show_all_cards"] = False
    except ApiError as exc:
        st.error(f"Couldn't build the teaching map: {exc}")

response = st.session_state.get("mindmap_response")
cards = st.session_state.get("educator_cards", [])

if response:
    st.success(
        f"Found {len(response['graph'].get('nodes', []))} connected teaching items "
        f"from {response.get('total_sources_queried', 0)} sources."
    )

    st.subheader("What we found for you")
    if not cards:
        st.caption("No evidence cards were generated for this map yet.")
    else:
        show_all = st.session_state.get("show_all_cards", False)
        visible_cards = cards if show_all else cards[:CARDS_SHOWN_BY_DEFAULT]

        grouped: dict[str, list[dict]] = {}
        for card in visible_cards:
            grouped.setdefault(card.get("entity_type", "Other"), []).append(card)
        ordered_keys = [key for key in ENTITY_LEGEND if key in grouped]
        ordered_keys += [key for key in grouped if key not in ENTITY_LEGEND]

        for key in ordered_keys:
            group_cards = grouped[key]
            friendly_label = ENTITY_LEGEND.get(key, (key, None))[0]
            st.markdown(f"##### {friendly_label} ({len(group_cards)})")
            for card in group_cards:
                with st.expander(card["label"]):
                    render_educator_node_card(cards, card["node_id"], advanced_mode=advanced_mode)

        if len(cards) > CARDS_SHOWN_BY_DEFAULT and not show_all:
            if st.button(f"Show all {len(cards)} items"):
                st.session_state["show_all_cards"] = True
                st.rerun()

    st.divider()
    with st.expander("🔍 Explore the connections visually (optional)", expanded=False):
        st.caption(
            "This shows the same items as a connected map instead of a list. "
            "Bubble color = item type, size = how strong the match is, and line "
            "style = how trustworthy the connection is."
        )
        render_trust_legend(response["graph"])
        selected = render_mindmap(response["graph"])
        if selected:
            labels = {
                node["id"]: node.get("label", selected)
                for node in response["graph"].get("nodes", [])
            }
            st.session_state["selected_node_id"] = selected
            st.success(
                f"You selected **{labels.get(selected, selected)}**. "
                "Find it above in the list, or expand \"Show all items\" if it's not visible yet."
            )

    if advanced_mode:
        with st.expander("Advanced agent steps"):
            st.json(response.get("agent_trace", []))
else:
    st.info(
        "Choose a guided workflow above, then select **Build teaching map**. "
        f"Suggested next step from your starter: "
        f"{(starter or {}).get('next_action', 'Open Curriculum Builder')}."
    )
