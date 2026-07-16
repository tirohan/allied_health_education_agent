from typing import Any

import streamlit as st
from streamlit_agraph import Config, Edge, Node, agraph

# Mirrors backend/app/agents/mindmap.py ENTITY_COLORS -- keep these two in sync
# so the legend always matches what's actually drawn.
_ENTITY_LEGEND: dict[str, tuple[str, str]] = {
    "Topic": ("Topic", "#4A90D9"),
    "Paper": ("Research paper", "#50C878"),
    "Resource": ("Teaching resource", "#F5A623"),
    "Program": ("Degree program", "#9B59B6"),
    "Competency": ("Competency", "#E74C3C"),
    "County": ("County", "#1ABC9C"),
    "ShortageArea": ("Shortage area", "#E67E22"),
    "SimulationCase": ("Simulation case", "#2ECC71"),
    "Institution": ("Institution", "#95A5A6"),
    "Discipline": ("Discipline", "#34495E"),
    "Author": ("Author", "#7F8C8D"),
}


def _swatch(color: str, text: str) -> str:
    return (
        f'<span style="display:inline-flex;align-items:center;margin-right:14px;'
        f'margin-bottom:4px;font-size:0.85rem;">'
        f'<span style="display:inline-block;width:11px;height:11px;border-radius:50%;'
        f'background:{color};margin-right:6px;flex-shrink:0;"></span>{text}</span>'
    )


def render_trust_legend(graph_data: dict[str, Any] | None = None) -> None:
    """Explain what the mind map's colors, sizes, and line styles mean.

    Without this, faculty see colored bubbles with no idea that color = item
    type, size = match strength, and line style = whether we could confirm the
    connection against the source database -- which is the whole point of the
    "verified" mind map.
    """
    present_types: set[str] | None = None
    if graph_data:
        present_types = {node.get("entity_type") for node in graph_data.get("nodes", [])}

    with st.expander("What am I looking at? (colors, sizes, lines explained)", expanded=False):
        st.markdown("**Bubble color = what kind of item it is**")
        entries = [
            entry
            for key, entry in _ENTITY_LEGEND.items()
            if present_types is None or key in present_types
        ]
        st.markdown(
            "".join(_swatch(color, label) for label, color in entries),
            unsafe_allow_html=True,
        )
        st.markdown("**Bubble size = how strong the match is.** Bigger means a stronger match to your question.")
        st.markdown("**Line style = how trustworthy the connection is:**")
        st.markdown(
            _swatch("#50C878", "Solid green — confirmed against our source database")
            + _swatch("#F5A623", "Dashed orange — AI-inferred, not yet directly confirmed")
            + _swatch("#95A5A6", "Dashed gray — not yet verified"),
            unsafe_allow_html=True,
        )
        st.caption(
            "Items that fail verification are automatically left off the map, "
            "so everything you see has at least some evidence behind it."
        )


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
