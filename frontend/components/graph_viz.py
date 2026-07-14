from typing import Any

import streamlit as st
from streamlit_agraph import Config, Edge, Node, agraph


def render_mindmap(graph_data: dict[str, Any]) -> str | None:
    nodes = [
        Node(
            id=node["id"],
            label=node["label"],
            size=node["size"],
            color=node["color"],
            title=node["tooltip"],
        )
        for node in graph_data.get("nodes", [])
    ]
    edges = [
        Edge(
            source=edge["source"],
            target=edge["target"],
            label=edge["label"],
            color=edge["color"],
            dashes=edge["dashes"],
            width=edge["weight"],
        )
        for edge in graph_data.get("edges", [])
    ]
    if not nodes:
        st.info("No graph nodes returned yet.")
        return None
    config = Config(
        width=950,
        height=650,
        directed=True,
        physics=True,
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor="#F7A7A6",
        collapsible=False,
    )
    return agraph(nodes=nodes, edges=edges, config=config)
