import streamlit as st

from frontend.components.guided_starters import render_guided_starters
from frontend.components.role_selector import render_role_selector

st.set_page_config(
    page_title="Allied Health Teaching Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Allied Health Teaching Intelligence")
st.write(
    "Plan courses, find trusted teaching materials, and connect education to "
    "local community and workforce needs."
)

with st.sidebar:
    st.header("Your role")
    render_role_selector()
    st.divider()
    advanced_mode = st.toggle(
        "Advanced / Research mode",
        value=st.session_state.get("advanced_mode", False),
        help="Show technical details such as agent steps and source table names.",
    )
    st.session_state["advanced_mode"] = advanced_mode

st.markdown(
    """
    ### What can you do here?
    1. Start from a teaching workflow instead of a blank search box.
    2. Explore a connected teaching map with plain language evidence.
    3. Build a curriculum outline, find local gaps, and export a teaching pack.
    """
)

starter = render_guided_starters()
if starter:
    st.success(
        f"Selected workflow: {starter['title']}. "
        "Open **Teaching Connections** in the sidebar to generate your map."
    )

st.markdown(
    """
    ### Pages
    - **Teaching Connections** — interactive planning map
    - **Evidence and Review** — why items are trustworthy, plus faculty feedback
    - **Find Teaching Materials** — search literature, resources, and cases
    - **Curriculum Builder** — printable teaching outline
    - **Gap Finder** — where shortage is high and teaching support is low
    - **Teaching Pack** — export a ready-to-share packet
    """
)

if st.session_state.get("teaching_list"):
    st.subheader("Your teaching list")
    for item in st.session_state["teaching_list"]:
        st.write(f"- {item.get('what_it_is')}: {item.get('label')}")
