import streamlit as st

from frontend.api_client import ApiError, post
from frontend.components.role_selector import render_role_selector

st.set_page_config(page_title="Curriculum Builder", layout="wide")
st.title("Curriculum Builder")
st.caption(
    "Turn a planning question into a printable outline: objectives, readings, "
    "simulation, and community context."
)

with st.sidebar:
    render_role_selector()

if st.session_state.pop("goto_page_hint", None) == "Curriculum Builder":
    st.caption("Opened from your teaching map selection.")

query = st.text_area(
    "Planning question",
    value=st.session_state.get(
        "planning_query",
        "Build an opioid IPE module for rural Georgia",
    ),
    height=100,
)

response = st.session_state.get("mindmap_response")
if not response:
    st.warning("No teaching map yet. Build one first, or generate one below.")
    if st.button("Generate teaching map for this question", type="primary"):
        try:
            with st.spinner("Building curriculum pathway..."):
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
                enrich = post(
                    "/api/v1/educator/enrich",
                    {
                        "query": query,
                        "role": st.session_state.get("educator_role"),
                        "graph": response["graph"],
                    },
                )
            st.session_state["mindmap_response"] = response
            st.session_state["educator_cards"] = enrich.get("cards", [])
            st.session_state["planning_query"] = query
            st.rerun()
        except ApiError as exc:
            st.error(f"Couldn't build the teaching map: {exc}")
    st.stop()

if st.button("Build curriculum outline", type="primary"):
    try:
        with st.spinner("Assembling learning pathway..."):
            outline = post(
                "/api/v1/educator/curriculum",
                {
                    "query": query,
                    "role": st.session_state.get("educator_role"),
                    "graph": response["graph"],
                },
            )
        st.session_state["curriculum_outline"] = outline
    except ApiError as exc:
        st.error(f"Couldn't build the curriculum outline: {exc}")

outline = st.session_state.get("curriculum_outline")
if not outline:
    st.info("Click Build curriculum outline to create a printable teaching plan.")
    st.stop()

st.subheader("Module title")
st.write(outline.get("title") or query)

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### Learning objectives")
    for item in outline.get("learning_objectives", []):
        st.markdown(f"- {item}")
    st.markdown("#### Recommended topic")
    st.write(outline.get("recommended_topic") or "—")
    st.markdown("#### Competencies")
    for item in outline.get("competencies", []) or ["—"]:
        st.markdown(f"- {item}")
    st.markdown("#### Teaching resources")
    for item in outline.get("teaching_resources", []) or ["—"]:
        st.markdown(f"- {item}")
with col2:
    st.markdown("#### Readings")
    for item in outline.get("readings", []) or ["—"]:
        st.markdown(f"- {item}")
    st.markdown("#### Simulation case")
    st.write(outline.get("simulation_case") or "None selected yet")
    st.markdown("#### Community context")
    for item in outline.get("community_context", []) or ["—"]:
        st.markdown(f"- {item}")
    st.markdown("#### Programs")
    for item in outline.get("programs", []) or ["—"]:
        st.markdown(f"- {item}")

st.markdown("#### Suggested sequence")
for item in outline.get("suggested_sequence", []) or []:
    st.markdown(f"- {item}")

st.markdown("#### Next teaching steps")
for item in outline.get("next_steps", []) or []:
    st.markdown(f"- {item}")

printable = outline.get("printable_markdown") or ""
if printable:
    st.download_button(
        "Download outline (Markdown)",
        data=printable,
        file_name="curriculum_outline.md",
        mime="text/markdown",
    )
    with st.expander("Printable outline preview"):
        st.markdown(printable)
