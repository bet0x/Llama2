import streamlit as st
from st_pages import Page, Section, show_pages, add_page_title
from st_pages import show_pages_from_config


st.set_page_config(
    page_title="Wikipedia",
     page_icon="https://api.dicebear.com/5.x/bottts-neutral/svg?seed=gptLAb"#,
)

add_page_title()

val = st.slider("This is a Slider", min_value=50, max_value=150, value=70)
print(val)

user_input = st.text_input("Enter your Course Title here")
st.write(user_input)

course_descr = st.text_area("Course Description")
st.write(course_descr)

user_date = st.date_input("Enter your registraion Date")
user_time = st.time_input("Set Timer")