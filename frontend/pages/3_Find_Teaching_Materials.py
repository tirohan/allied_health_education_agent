import pandas as pd
import streamlit as st

from frontend.api_client import ApiError, post
from frontend.components.role_selector import render_role_selector

st.set_page_config(page_title="Find Teaching Materials", layout="wide")
st.title("Find Teaching Materials")
st.caption(
    "Search research articles, teaching resources, programs, counties, and simulation cases."
)

with st.sidebar:
    render_role_selector()
    collection_labels = {
        "papers": "Research articles",
        "resources": "Teaching resources",
        "programs": "Degree programs",
        "communities": "Georgia counties",
        "simulation_cases": "Simulation cases",
    }
    collections = st.multiselect(
        "Search in",
        options=list(collection_labels.keys()),
        default=["papers", "resources", "simulation_cases"],
        format_func=lambda key: collection_labels[key],
    )
    top_k = st.slider("Number of results", 5, 40, 12)

query = st.text_input(
    "What do you want to teach or find?",
    value=st.session_state.get("planning_query", "opioid interprofessional education"),
)

if st.button("Search materials", type="primary"):
    try:
        with st.spinner("Searching trusted education sources..."):
            response = post(
                "/api/v1/search",
                {
                    "query": query,
                    "collections": collections,
                    "top_k": top_k,
                    "filters": st.session_state.get("planning_filters", {}),
                },
            )
        st.session_state["search_response"] = response
    except ApiError as exc:
        st.error(f"Search failed: {exc}")

response = st.session_state.get("search_response")
if response is not None and not response.get("results"):
    st.info("No matching materials found. Try a broader query or different sources.")
elif response:
    type_map = {
        "papers": "Research article",
        "resources": "Teaching resource",
        "programs": "Degree program",
        "communities": "County context",
        "simulation_cases": "Simulation case",
    }
    rows = [
        {
            "What it is": type_map.get(result["collection"], result["collection"]),
            "Title": result["title"],
            "Relevance score": round(float(result["score"]), 3),
            "Where it came from": result["source_table"].replace("_", " "),
            "What to do next": "Open Curriculum Builder or Save to teaching list",
        }
        for result in response.get("results", [])
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    st.info(
        "Tip: use Teaching Connections if you want these materials connected into a planning map."
    )
