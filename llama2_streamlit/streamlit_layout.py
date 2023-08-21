import streamlit as st

## Streamlit Elements
# pip install streamlit-elements==0.1.*
from streamlit_elements import elements, mui, html

with elements("new element"):
    mui.Typography("Hello World")


## Streamlit Pages
# pip install st-pages

from st_pages import Page, Section, show_pages, add_page_title

add_page_title()

show_pages(
    [
                Page("streamlit_form.py","Introduction","🏠"),
                Page("streamlit_sidebar.py","Content",":books:"),
                Section("My section", icon="🎈️"),

                Page("streamlit_write_magic.py", icon="💪"),
                Page("streamlit_dataframe.py", in_section=False)
            ]
    )

## Sidebar
st.sidebar.write("This live in the sidebar")
st.sidebar.button("CLick Me")

## Columns
col1, col2 = st.columns(2)
col1.write("This is column 1")
col2.write("This is column 2 ")

## Tabs
tab1, tab2 = st.tabs(["Tab 1", "Tab 2"])
tab1.write("This is Tab 1")
tab2.write("This is Tab 2")

## Expander
with st.expander("Open to see the answers"):
    st.write("This is details response of it")

## Container
c = st.container()
st.write("This will show last")
c.write("This will show first ")
c.write("This will show second")

## Empty
ce = st.empty()
st.write("This will show last")
ce.write("This will be replaced")
ce.write("THis will show first")

