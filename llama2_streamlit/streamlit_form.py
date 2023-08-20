import streamlit as st
import pandas as pd
from time import sleep
from datetime import time

# To remove streamlit footer
st.markdown("""
<style>
.css-cio0dv.ea3mdgi1
{
    visibility:hidden;
} 
</style>             
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'> User Registration</h1>", unsafe_allow_html=True)

# Create a form for this specific form
form = st.form("Form 1")
form.text_input("First Name Form 1")
form.form_submit_button("Submit")

with st.form("Form 2"):
    col1,col2 = st.columns(2)
    f_name = col1.text_input("First Name")
    l_name = col2.text_input("Last Name")
    st.text_input("Email Address")
    st.text_input("Password")
    st.text_input("Confirm Password")
    day,month,year = st.columns(3)
    day.text_input("Day")
    month.text_input("Month")
    year.text_input("Year")
    st.form_submit_button("Submit")
    
    if st:
        if f_name == "" and l_name=="":
            st.warning("Please fill the field")
        else:
            st.success("Submitted Succesffully")
        
