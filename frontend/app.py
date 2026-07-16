import streamlit as st

from frontend.components.guided_starters import render_guided_starters
from frontend.components.role_selector import render_role_selector
from frontend.components.teaching_list import render_teaching_list_sidebar

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

render_teaching_list_sidebar()

st.markdown(
    """
    ### What can you do here?
    1. Start from a teaching workflow instead of a blank search box.
    2. Explore a connected teaching map with plain language evidence.
    3. Build a curriculum outline, find local gaps, and export a teaching pack.
    """
)

with st.expander("Why should I trust what this tool shows me?", expanded=True):
    st.markdown(
        """
Every item on your teaching map is checked against our source database before you
ever see it -- this isn't just an AI guessing:

- **✅ Confirmed** — directly matched to a real record in our database
- **🟠 AI-inferred** — a likely connection the AI made, not yet directly confirmed
- **⚪ Unverified** — not yet checked, so treat it with extra caution
- Anything that fails verification is automatically left off the map

You'll see these labels (and a full legend) on every teaching map, and a
trust summary on the **Evidence and Review** page.
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
