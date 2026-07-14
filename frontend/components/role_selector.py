import streamlit as st

from frontend.api_client import get


DEFAULT_ROLES = [
    {"id": "pt_faculty", "label": "PT faculty member"},
    {"id": "ot_faculty", "label": "OT faculty member"},
    {"id": "nursing_faculty", "label": "Nursing faculty member"},
    {"id": "simulation_director", "label": "Simulation director"},
    {"id": "program_chair", "label": "Program chair"},
    {"id": "researcher", "label": "Researcher"},
]


def render_role_selector() -> str:
    try:
        roles = get("/api/v1/educator/roles").get("roles", DEFAULT_ROLES)
    except Exception:
        roles = DEFAULT_ROLES
    labels = {role["id"]: role["label"] for role in roles}
    choice = st.selectbox(
        "I am a...",
        options=list(labels.keys()),
        format_func=lambda key: labels[key],
        index=list(labels.keys()).index(st.session_state.get("educator_role", "pt_faculty"))
        if st.session_state.get("educator_role", "pt_faculty") in labels
        else 0,
        help="This adapts the language and next actions to your teaching role.",
    )
    st.session_state["educator_role"] = choice
    return choice
