import streamlit as st
from st_pages import Page, Section, show_pages, add_page_title
from st_pages import show_pages_from_config

add_page_title()

# st.title("Objectives")
st.markdown( 
    """
    ### <span style="color:green"> Integration of AI and Machine Learning </span>
    * To explore the integration of AI and ML techniques and implement AI-drive solutions that able to handle the arising number of tickets effectively.

    ### <span style="color:green"> Wiki Knowledge based </span>
    * To create general knowledge base serves as centralized repository of information, encompassing of a wide range of topics and solutions relevant to customers for example `“Vector Database”` supported by AI powered search algorithm that provide fast and accurate retrieval of relevant information from the base 

    ### <span style="color:green"> Smart Question and Answer System </span>
    * The need to **`develop`** a method to identify and extract the most commonly asked question from ticket data.
    
    """
,unsafe_allow_html=True)



