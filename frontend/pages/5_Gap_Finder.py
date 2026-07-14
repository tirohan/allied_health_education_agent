import pandas as pd
import streamlit as st

from frontend.api_client import post
from frontend.components.role_selector import render_role_selector

st.set_page_config(page_title="Gap Finder", layout="wide")
st.title("Gap Finder")
st.caption(
    "Compare shortage severity with nearby programs and available teaching materials. "
    "Focus on what is missing."
)

with st.sidebar:
    render_role_selector()

topic = st.text_input(
    "Topic",
    value=st.session_state.get("gap_topic", "opioid"),
)
county = st.text_input(
    "County (optional)",
    value=st.session_state.get("gap_county", ""),
    placeholder="e.g. Coffee County",
)

if st.button("Find teaching gaps", type="primary"):
    with st.spinner("Comparing shortages, programs, and teaching materials..."):
        result = post(
            "/api/v1/educator/gaps",
            {
                "topic": topic,
                "topic_keywords": topic,
                "county": county or None,
                "state": "GA",
                "limit": 20,
            },
        )
        st.session_state["gap_result"] = result
        st.session_state["gap_topic"] = topic
        st.session_state["gap_county"] = county

result = st.session_state.get("gap_result")
if not result:
    st.info("Enter a topic (and optional county) to see where teaching support is thin.")
    st.stop()

summary = result.get("summary") or result.get("educator_summary") or ""
if summary:
    st.success(summary)

metrics = result.get("metrics") or {}
if metrics:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Shortage signal", metrics.get("shortage_severity", "—"))
    m2.metric("Nearby programs", metrics.get("nearby_programs", "—"))
    m3.metric("OER / resources", metrics.get("available_resources", "—"))
    m4.metric("Simulation cases", metrics.get("simulation_cases", "—"))

gaps = result.get("gaps") or []
if gaps:
    st.subheader("What is missing")
    st.dataframe(pd.DataFrame(gaps), use_container_width=True)

recommendations = result.get("recommendations") or []
if recommendations:
    st.subheader("What to do next")
    for item in recommendations:
        st.markdown(f"- {item}")

signal = result.get("statewide_teaching_signal") or {}
if signal and st.session_state.get("advanced_mode"):
    with st.expander("Advanced teaching supply counts"):
        st.json(signal)
