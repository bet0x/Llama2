import streamlit as st
from st_pages import Page, Section, show_pages, add_page_title
from st_pages import show_pages_from_config

st.set_page_config(
    page_title="Wikipedia",
     page_icon="https://api.dicebear.com/5.x/bottts-neutral/svg?seed=gptLAb"#,
)

add_page_title()

images = st.file_uploader("Please upload an image", type=["png","jpg"], accept_multiple_files=True)
if images is not None:
    for image in images:
        st.image(image)
        
