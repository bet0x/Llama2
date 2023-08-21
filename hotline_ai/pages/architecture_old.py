import streamlit as st
from st_pages import Page, Section, show_pages, add_page_title
from st_pages import show_pages_from_config

add_page_title()

st.title("Wiki v1.0 Backend Architecture")
st.markdown( 
    """
    ### Elastic Database
    * Central repository used to store wiki information.
    * Serve as a Knowledge Base.

    ### Source Data
    * Information collected from Tim (First FAQ).
    * This data was pre-processed and converted to `json`.
    * This data push and store into Elastic db.

    ### Search Mechanism
    * Data are queries based on pattern, regex and logical function matching.

    """
)

st.image("./images/wiki_v1.jpg", caption="Fig 1: Wiki System Architecture Wiki 1.0")
st.image("./images/wiki_v1_con.jpg", caption="Fig 2: Basic Pattern Matching")

