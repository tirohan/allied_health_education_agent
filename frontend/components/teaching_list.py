import streamlit as st


def render_teaching_list_sidebar() -> None:
    """Always-visible sidebar widget so saved items are never out of sight.

    The list lives in st.session_state, so it resets when the browser session
    ends -- callers should not assume it survives a refresh or a new visit.
    """
    teaching_list = st.session_state.get("teaching_list", [])
    with st.sidebar:
        st.divider()
        st.subheader(f"Your teaching list ({len(teaching_list)})")
        if not teaching_list:
            st.caption("Items you save from a teaching map will show up here.")
            return
        st.caption("Saved for this browser session. Export a teaching pack before you close the tab.")
        for item in teaching_list:
            node_id = item.get("node_id")
            cols = st.columns([0.82, 0.18])
            with cols[0]:
                st.markdown(f"**{item.get('label', 'Untitled item')}**")
                st.caption(item.get("what_it_is", ""))
            with cols[1]:
                if st.button("✕", key=f"remove_teaching_list_{node_id}", help="Remove from list"):
                    st.session_state["teaching_list"] = [
                        entry for entry in teaching_list if entry.get("node_id") != node_id
                    ]
                    st.rerun()
