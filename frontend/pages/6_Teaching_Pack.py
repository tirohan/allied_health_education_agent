import json

import streamlit as st

from frontend.api_client import ApiError, post, post_bytes
from frontend.components.role_selector import render_role_selector

st.set_page_config(page_title="Teaching Pack", layout="wide")
st.title("Teaching Pack Export")
st.caption(
    "Export a short packet for class prep: resources, papers, a simulation case, "
    "county indicators, and citations."
)

with st.sidebar:
    render_role_selector()
    export_format = st.selectbox("Export format", ["docx", "markdown", "json"])

query = st.text_area(
    "Pack focus",
    value=st.session_state.get(
        "planning_query",
        "Build an opioid IPE module for rural Georgia",
    ),
    height=90,
)

response = st.session_state.get("mindmap_response")
if not response:
    st.warning("Build a teaching map first so the pack has connected materials.")
    if st.button("Generate teaching map", type="primary"):
        try:
            with st.spinner("Gathering materials..."):
                response = post(
                    "/api/v1/mindmap",
                    {
                        "query": query,
                        "max_nodes": 30,
                        "min_confidence": 0.45,
                        "collections": st.session_state.get(
                            "planning_collections",
                            [
                                "papers",
                                "resources",
                                "programs",
                                "communities",
                                "simulation_cases",
                            ],
                        ),
                        "filters": st.session_state.get("planning_filters", {"state": "GA"}),
                    },
                )
            st.session_state["mindmap_response"] = response
            st.session_state["planning_query"] = query
            st.rerun()
        except ApiError as exc:
            st.error(f"Couldn't build the teaching map: {exc}")
    st.stop()

if st.button("Build teaching pack", type="primary"):
    try:
        with st.spinner("Assembling teaching pack..."):
            pack = post(
                "/api/v1/educator/teaching-pack",
                {
                    "query": query,
                    "role": st.session_state.get("educator_role"),
                    "graph": response["graph"],
                    "format": "json",
                },
            )
        st.session_state["teaching_pack"] = pack
    except ApiError as exc:
        st.error(f"Couldn't build the teaching pack: {exc}")

pack = st.session_state.get("teaching_pack")
if not pack:
    st.info("Click Build teaching pack to preview and download.")
    st.stop()

st.subheader(pack.get("title") or "Teaching pack")
st.write(
    f"Role: {pack.get('role', 'Educator')} · "
    f"Question: {pack.get('planning_question', query)}"
)

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### Resources (up to 5)")
    for item in pack.get("resources", [])[:5] or ["—"]:
        st.markdown(f"- {item}")
    st.markdown("#### Papers (up to 2)")
    for item in pack.get("papers", [])[:2] or ["—"]:
        st.markdown(f"- {item}")
with col2:
    st.markdown("#### Simulation case")
    st.write(pack.get("simulation_case") or "—")
    st.markdown("#### County indicators")
    for item in pack.get("county_indicators", []) or ["—"]:
        st.markdown(f"- {item}")

st.markdown("#### How to use")
for item in pack.get("how_to_use", []) or []:
    st.markdown(f"- {item}")

st.markdown("#### Citations")
for citation in pack.get("citations", []) or []:
    if isinstance(citation, dict):
        st.markdown(
            f"- {citation.get('label')} ({citation.get('type')}, "
            f"{citation.get('evidence')})"
        )
    else:
        st.markdown(f"- {citation}")

st.subheader("Download")
if export_format == "json":
    st.download_button(
        "Download JSON",
        data=json.dumps(pack, indent=2).encode("utf-8"),
        file_name="teaching_pack.json",
        mime="application/json",
    )
else:
    try:
        with st.spinner(f"Preparing {export_format.upper()} export..."):
            content, content_type = post_bytes(
                "/api/v1/educator/teaching-pack",
                {
                    "query": query,
                    "role": st.session_state.get("educator_role"),
                    "graph": response["graph"],
                    "format": export_format,
                },
            )
        ext = "docx" if export_format == "docx" else "md"
        st.download_button(
            f"Download {export_format.upper()}",
            data=content,
            file_name=f"teaching_pack.{ext}",
            mime=content_type,
        )
    except ApiError as exc:
        st.error(f"Export failed: {exc}")
