import altair as alt
import pandas as pd
import streamlit as st

from frontend.api_client import ApiError, post
from frontend.components.role_selector import render_role_selector
from frontend.components.teaching_list import render_teaching_list_sidebar

st.set_page_config(page_title="Gap Finder", layout="wide")
st.title("Gap Finder")
st.caption(
    "Compare shortage severity with nearby programs and available teaching materials. "
    "Focus on what is missing."
)

with st.sidebar:
    render_role_selector()

render_teaching_list_sidebar()

if st.session_state.pop("goto_page_hint", None) == "Gap Finder":
    st.caption("Opened from your teaching map selection.")

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
    try:
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
    except ApiError as exc:
        st.error(f"Couldn't find teaching gaps: {exc}")

result = st.session_state.get("gap_result")
if not result:
    st.info("Enter a topic (and optional county) to see where teaching support is thin.")
    st.stop()

summary = result.get("summary") or result.get("educator_summary") or ""
if summary:
    st.success(summary)

metrics = result.get("metrics") or {}
if metrics:
    severity = metrics.get("shortage_severity", "Unknown")
    severity_icon = {"High": "🔴", "Moderate": "🟠"}.get(severity, "⚪")
    resources = metrics.get("available_resources")
    cases = metrics.get("simulation_cases")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        f"{severity_icon} Shortage signal",
        severity,
        help="Based on federal health professional shortage area (HPSA) designations for this county.",
    )
    m2.metric(
        "Nearby programs",
        metrics.get("nearby_programs", "—"),
        help="Allied health degree programs in Georgia institutions matching this topic.",
    )
    m3.metric(
        "OER / resources",
        resources if resources is not None else "—",
        delta="Below typical coverage" if isinstance(resources, int) and resources < 25 else None,
        delta_color="inverse",
        help="Teaching resources matching this topic. Fewer than ~25 is thin coverage for a full module.",
    )
    m4.metric(
        "Simulation cases",
        cases if cases is not None else "—",
        delta="Limited — consider new case design" if isinstance(cases, int) and cases < 10 else None,
        delta_color="inverse",
        help="Simulation cases matching this topic. Fewer than ~10 leaves little variety to draw from.",
    )

gaps = result.get("gaps") or []
county_gaps = result.get("county_gaps") or []
if len(county_gaps) > 1:
    st.subheader("Where is the gap? (top counties by shortage severity)")
    chart_df = pd.DataFrame(county_gaps)
    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("max_hpsa_score:Q", title="Shortage severity (HPSA score)"),
            y=alt.Y("county_name:N", sort="-x", title="County"),
            color=alt.Color(
                "gap_level:N",
                title="Gap level",
                scale=alt.Scale(
                    domain=["High", "Moderate", "Lower"],
                    range=["#E74C3C", "#F5A623", "#95A5A6"],
                ),
            ),
            tooltip=[
                alt.Tooltip("county_name:N", title="County"),
                alt.Tooltip("gap_level:N", title="Gap level"),
                alt.Tooltip("max_hpsa_score:Q", title="HPSA score"),
                alt.Tooltip("poverty_percentage:Q", title="Poverty %", format=".1f"),
            ],
        )
        .properties(height=max(200, 26 * len(chart_df)))
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption(
        "We don't have county map coordinates in the database, so this ranks counties "
        "by shortage severity instead of plotting them geographically."
    )

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
