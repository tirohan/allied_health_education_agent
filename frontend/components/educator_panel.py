from typing import Any

import streamlit as st

from frontend.api_client import post


def _card_for_node(
    cards: list[dict[str, Any]],
    node_id: str | None,
) -> dict[str, Any] | None:
    if not node_id:
        return None
    return next((card for card in cards if card.get("node_id") == node_id), None)


def render_educator_node_card(
    cards: list[dict[str, Any]],
    selected_node_id: str | None,
    advanced_mode: bool = False,
) -> None:
    card = _card_for_node(cards, selected_node_id)
    if card is None:
        st.info(
            "Select an item in the map to see what it is, why it appeared, "
            "who it is for, and what to do next."
        )
        return

    st.subheader(card["label"])
    st.markdown(f"**What it is:** {card['what_it_is']}")
    st.markdown(f"**Why it appeared for this question:** {card['why_it_appeared']}")
    st.markdown(f"**Who it is for:** {card['who_it_is_for']}")
    st.markdown(f"**How strong the evidence is:** {card['evidence_strength']}")
    st.success(card["evidence_plain"])

    st.markdown("**What you can do next**")
    action_cols = st.columns(min(3, max(1, len(card["next_actions"]))))
    for index, action in enumerate(card["next_actions"]):
        with action_cols[index % len(action_cols)]:
            if st.button(action, key=f"action_{card['node_id']}_{index}"):
                st.session_state["last_educator_action"] = {
                    "action": action,
                    "node_id": card["node_id"],
                    "label": card["label"],
                }
                if action == "Save to teaching list":
                    teaching_list = st.session_state.setdefault("teaching_list", [])
                    if card["node_id"] not in {item.get("node_id") for item in teaching_list}:
                        teaching_list.append(card)
                        st.toast("Saved to your teaching list.")
                elif action in {"Open Curriculum Builder", "Use in syllabus", "Map to resources"}:
                    st.session_state["goto_page_hint"] = "Curriculum Builder"
                    st.toast("Open Curriculum Builder from the sidebar to turn this into a syllabus outline.")
                elif action in {"Compare counties", "Open Gap Finder"}:
                    st.session_state["goto_page_hint"] = "Gap Finder"
                    if card.get("label"):
                        st.session_state["gap_county"] = card["label"]
                    st.toast("Open Gap Finder from the sidebar to compare shortage vs teaching support.")
                elif action == "Open resource":
                    if card.get("source_url"):
                        st.link_button("Open original source", card["source_url"])
                    else:
                        st.toast("No public URL is stored for this item yet.")

    if st.session_state.get("last_educator_action"):
        last = st.session_state["last_educator_action"]
        st.caption(f"Last action: {last['action']} on {last['label']}")

    st.markdown("**Faculty review**")
    decision = st.radio(
        "Is this useful for teaching?",
        ["Useful", "Not relevant", "Needs review"],
        horizontal=True,
        key=f"review_decision_{card['node_id']}",
    )
    notes = st.text_input(
        "Optional note",
        key=f"review_notes_{card['node_id']}",
        placeholder="Example: Good for first year IPE seminar",
    )
    if st.button("Submit review", key=f"review_submit_{card['node_id']}"):
        advanced = card.get("advanced") or {}
        result = post(
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
        with st.expander("Advanced / Research mode"):
            st.json(
                {
                    "entity_type": card.get("entity_type"),
                    "verification_status": card.get("verification_status"),
                    "confidence": card.get("confidence"),
                    "source_plain": card.get("source_plain"),
                    "advanced": card.get("advanced"),
                }
            )


def render_plain_evidence_list(cards: list[dict[str, Any]]) -> None:
    st.subheader("Why you can trust these items")
    if not cards:
        st.caption("Generate a teaching map to see plain language evidence.")
        return
    for card in cards[:12]:
        st.markdown(f"**{card['label']}**")
        st.caption(card["evidence_plain"])
