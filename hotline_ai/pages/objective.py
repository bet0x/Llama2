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
    * To establish an intelligent system that can understand the user question and provide relevant responses or suggestion. This AI powered self-service allow customer to find answers to their question independently, empowering them to resolve issues without waiting for human support and provide consistent , updated and accurate responses based on the question, reducing risk of human error and ensuring uniform support quality.
    * Routine and repetitive inquires can be handles by the AI system, allowing human agents to focus on more complex and high-touch customer interactions which reduce the ticket volume.
    * Time and cost savings due to reduced supports costs and increased agent productivity as fewer resources are required to address routine inquiries which lead to reduce number of tickets.

    
    """
,unsafe_allow_html=True)



