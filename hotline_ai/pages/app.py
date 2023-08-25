import streamlit as st
from st_pages import Page, Section, show_pages, add_page_title
from st_pages import show_pages_from_config


st.set_page_config(
    page_title="Wikipedia",
     page_icon="https://api.dicebear.com/5.x/bottts-neutral/svg?seed=gptLAb"#,
)
try:
    add_page_title()

    st.title("Hotline AI and Future of CTS")
    st.markdown("### Feasibility :green[Study] of Hotline AI and Future of CTS")

    st.markdown(
        """

        In this study, our primary emphasis lies in assessing the feasibility of integrating AI
        into hotline tasks. Our aim is to enhance the efficiency of customer support and establish
        a comprehensive internal knowledge base. The study centers around identifying the most effective
        and reliable approach to extract accurate responses from the AI model, ensuring optimal outcomes
        for addressing customer inquiries.
    
        **👈 Select a demo from the sidebar** to see the differences of difference approach !

        ### Want to learn more?
        - Check our [documentation](https://docs.streamlit.io)
        - Addres your question to [Hotline](hotline@xfab.com)
        
        ### Developed By
        - [Lukas Johnny](https://github.com/streamlit/demo-self-driving)
        - [Github](https://github.com/streamlit/demo-self-driving)
    """
    )

    show_pages_from_config("./pages/.streamlit/pages_sections.toml")
except:
    pass


