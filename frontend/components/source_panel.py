from typing import Any

import streamlit as st


def render_node_details(graph_data: dict[str, Any], selected_node_id: str | None) -> None:
    if not selected_node_id:
        st.caption("Select a node to inspect provenance.")
        return
    node = next((item for item in graph_data.get("nodes", []) if item["id"] == selected_node_id), None)
    if node is None:
        st.warning("Selected node was not found in the graph payload.")
        return
    st.subheader(node["label"])
    st.write(f"Type: `{node['entity_type']}`")
    st.write(f"Status: `{node['verification_status']}`")
    st.write(f"Confidence: `{node['confidence']:.2f}`")
    st.write(f"Source: `{node['source_table']}.{node['source_id']}`")
    st.caption(node.get("tooltip", ""))


def render_citations(citations: list[dict[str, Any]]) -> None:
    st.subheader("Citations")
    if not citations:
        st.caption("No citations returned yet.")
        return
    for citation in citations:
        label = citation.get("label") or citation.get("source_id")
        st.markdown(f"**{label}**")
        st.caption(f"{citation.get('source_table')}.{citation.get('source_id')}")
        if citation.get("doi"):
            st.caption(f"DOI: {citation['doi']}")
        if citation.get("url"):
            st.link_button("Open Source", citation["url"])
        if citation.get("evidence_snippet"):
            st.write(citation["evidence_snippet"])
