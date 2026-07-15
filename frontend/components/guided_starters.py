from typing import Any

import streamlit as st

from frontend.api_client import ApiError, get


FALLBACK_STARTERS = [
    {
        "id": "opioid_ipe_rural_ga",
        "title": "Build an opioid IPE module for rural Georgia",
        "description": "Connect topics, teaching resources, simulation cases, and rural county context.",
        "query": (
            "What interprofessional education resources address opioid education "
            "in rural Georgia counties?"
        ),
        "collections": ["papers", "resources", "programs", "communities", "simulation_cases"],
        "filters": {"state": "GA"},
        "next_action": "Open Curriculum Builder",
    },
    {
        "id": "community_sim_cases",
        "title": "Find simulation cases grounded in local community data",
        "description": "Link simulation cases to Georgia county indicators and shortage context.",
        "query": (
            "Which simulation cases support community based allied health training "
            "for Georgia counties with local health needs?"
        ),
        "collections": ["simulation_cases", "communities", "resources"],
        "filters": {"state": "GA"},
        "next_action": "Compare counties",
    },
    {
        "id": "programs_near_shortages",
        "title": "See which allied health programs exist near shortage counties",
        "description": "Explore program availability around high need Georgia counties.",
        "query": (
            "Which allied health programs are available near Georgia health "
            "professional shortage counties?"
        ),
        "collections": ["programs", "communities"],
        "filters": {"state": "GA"},
        "next_action": "Open Gap Finder",
    },
    {
        "id": "competency_oer_map",
        "title": "Map competencies to available open educational resources",
        "description": "Find OER and agency resources that support competency based teaching.",
        "query": (
            "Which open educational resources address interprofessional collaboration "
            "and substance use competencies for allied health learners?"
        ),
        "collections": ["resources", "papers"],
        "filters": {},
        "next_action": "Save to teaching list",
    },
    {
        "id": "gap_shortage_resources",
        "title": "Identify gaps: shortage high, teaching resources low",
        "description": "Highlight places where workforce need is high but teaching materials are thin.",
        "query": (
            "Where do Georgia shortage counties have high need but limited opioid "
            "or behavioral health teaching resources?"
        ),
        "collections": ["communities", "resources", "programs", "simulation_cases"],
        "filters": {"state": "GA"},
        "next_action": "Open Gap Finder",
    },
]


def load_starters() -> list[dict[str, Any]]:
    try:
        return get("/api/v1/educator/starters").get("starters", FALLBACK_STARTERS)
    except ApiError:
        st.caption("⚠️ Using default workflows — couldn't reach the server.")
        return FALLBACK_STARTERS


def render_guided_starters() -> dict[str, Any] | None:
    starters = load_starters()
    st.subheader("Start with a teaching workflow")
    st.caption("Choose a planning goal. The system will build a connected teaching map for you.")
    selected = None
    for starter in starters:
        cols = st.columns([0.78, 0.22])
        with cols[0]:
            st.markdown(f"**{starter['title']}**")
            st.caption(starter["description"])
        with cols[1]:
            if st.button("Use this", key=f"starter_{starter['id']}"):
                selected = starter
                st.session_state["selected_starter"] = starter
                st.session_state["planning_query"] = starter["query"]
                st.session_state["planning_collections"] = starter["collections"]
                st.session_state["planning_filters"] = starter.get("filters", {})
    return selected or st.session_state.get("selected_starter")
