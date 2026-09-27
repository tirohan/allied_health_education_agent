import pandas as pd
import streamlit as st

from frontend.api_client import ApiError, post, safe_call
from frontend.components.educator_panel import render_plain_evidence_list, render_record_evidence
from frontend.components.role_selector import render_role_selector
from frontend.components.teaching_list import render_teaching_list_sidebar

st.set_page_config(page_title="Evidence and Review", layout="wide")
st.title("Evidence and Review")
st.caption(
    "See how each item was checked, then mark it as useful, not relevant, or needing review."
)

with st.sidebar:
    render_role_selector()
    advanced_mode = st.toggle(
        "Advanced / Research mode",
        value=st.session_state.get("advanced_mode", False),
    )
    st.session_state["advanced_mode"] = advanced_mode

render_teaching_list_sidebar()

response = st.session_state.get("mindmap_response")
cards = st.session_state.get("educator_cards", [])

if not response:
    st.info("Build a teaching map first from the Teaching Connections page.")
    st.stop()

if not cards:
    try:
        cards = safe_call(
            "Loading evidence...",
            post,
            "/api/v1/educator/enrich",
            {
                "query": st.session_state.get(
                    "planning_query", response["graph"].get("query", "")
                ),
                "role": st.session_state.get("educator_role"),
                "graph": response["graph"],
            },
        ).get("cards", [])
        st.session_state["educator_cards"] = cards
    except ApiError as exc:
        st.error(f"Couldn't load evidence: {exc}")
        st.stop()

graph = response.get("graph") or {}
st.caption(f"Data snapshot: {graph.get('snapshot_id') or 'unsnapshotted'}")
for note in graph.get("scope_notes") or []:
    st.info(note)

STATUS_ORDER = ["CONFIRMED", "INFERRED", "UNVERIFIED", "CONTESTED", "REFUTED"]
STATUS_HELP = {
    "CONFIRMED": "Checked against a source database record (for paper-topic links: a human-validated mapping).",
    "INFERRED": "A plausible link that is NOT confirmed fact. Treat as a lead for expert review.",
    "UNVERIFIED": "No supporting source record was found yet. Treat with extra caution.",
    "CONTESTED": "The automated critic judged the link contradicted by its own evidence. It can only downgrade, never confirm.",
    "REFUTED": "Contradicted by the database or rejected by faculty review; hidden from the map.",
}
STATUS_LABEL = {
    "CONFIRMED": "✅ Confirmed", "INFERRED": "🟠 Inferred", "UNVERIFIED": "⚪ Unverified",
    "CONTESTED": "🟣 Contested", "REFUTED": "🔴 Refuted",
}


def _status_counts(statuses) -> dict[str, int]:
    counts = dict.fromkeys(STATUS_ORDER, 0)
    for status in statuses:
        if status in counts:  # edges from pre-revision cached maps carry no status
            counts[status] += 1
    return counts


if cards:
    st.markdown("**How was this map checked, at a glance?**")
    for title, counts in (
        ("Items", _status_counts(card.get("verification_status") for card in cards)),
        ("Connections", _status_counts(edge.get("verification_status") for edge in graph.get("edges", []))),
    ):
        st.caption(title)
        for column, status in zip(st.columns(len(STATUS_ORDER)), STATUS_ORDER):
            column.metric(STATUS_LABEL[status], counts[status], help=STATUS_HELP[status])
    st.caption(
        "Only Confirmed items were checked against a source record. Inferred and "
        "Unverified items are shown for review and are not confirmed facts; Contested "
        "connections were flagged by the automated critic; Refuted items are hidden from the map."
    )

render_plain_evidence_list(cards)

st.subheader("Review queue for faculty")
if not cards:
    st.info("No evidence cards are available for this teaching map yet.")
    st.stop()

rows = [
    {
        "Item": card["label"],
        "What it is": card["what_it_is"],
        "Evidence": card["evidence_strength"],
        "Next actions": ", ".join(card["next_actions"]),
    }
    for card in cards
]
st.dataframe(pd.DataFrame(rows), use_container_width=True)

st.subheader("Submit a faculty decision")
options = {card["node_id"]: card["label"] for card in cards}
selected_id = st.selectbox(
    "Choose an item",
    options=list(options.keys()),
    format_func=lambda key: options[key],
)
card = next((item for item in cards if item["node_id"] == selected_id), None)
if card is None:
    st.info("Select an item above to submit a faculty decision.")
    st.stop()

render_record_evidence(card)

decision = st.radio(
    "Faculty decision",
    ["Useful", "Not relevant", "Needs review"],
    horizontal=True,
)
notes = st.text_area("Notes for the review team")
if st.button("Save faculty review", type="primary"):
    advanced = card.get("advanced") or {}
    result = safe_call(
        "Saving your review...",
        post,
        "/api/v1/educator/review",
        {
            "record_type": advanced.get("source_table") or card["entity_type"],
            "record_id": advanced.get("source_id") or card["node_id"],
            "record_title": card["label"],
            "decision": decision,
            "reviewer": st.session_state.get("educator_role", "faculty_user"),
            "notes": notes or None,
        },
    )
    st.success(result.get("message", "Review saved."))

if advanced_mode:
    with st.expander("Advanced verification details"):
        st.json(
            [
                {
                    "label": card["label"],
                    "verification_status": card["verification_status"],
                    "confidence": card["confidence"],
                    "advanced": card.get("advanced"),
                }
                for card in cards
            ]
        )
