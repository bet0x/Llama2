import streamlit as st
from st_pages import Page, Section, show_pages, add_page_title
from st_pages import show_pages_from_config

add_page_title()

# st.title("Problem Statement ✏️")
st.markdown( 
    """
    ### <span style="color:red"> Challenge of Ticket Volume and agent resources </span>
    * Dealing with an increasing number of tickets with limited Hotline agent support can be very limiting.
    * The need to have a system that enable customer to resolve straightforward queries on their own, reduces numbers of ticket and load on support agent allowing them to focus on more complex issues.


    ### <span style="color:red"> Data Facilitation and Analysis </span>
    * The need to organize, store and retrieve the ticket data effectively is becoming more crucial.
    * Finding way to efficiently analyze and process the data within the tickets can be challenging.

    ### <span style="color:red"> Identifying Most Frequently Asked Question (FAQs) </span>
    * The need to develop a method to identify and extract the most commonly asked question from ticket data.

    ### <span style="color:red"> Enhancing Customer Experience </span>
    * To have a system that able to contribute to an improved customer experience by providing a quick, consistent and accurate solutions while hotline agent managing the increased ticket workload. Customer can quickly access a repository of FAQ and find their corresponding answers, reducing time to contact for support.

    """    
,unsafe_allow_html=True)




