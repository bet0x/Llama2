import streamlit as st
from st_pages import Page, Section, show_pages, add_page_title
from st_pages import show_pages_from_config

add_page_title()

st.markdown("## Pending Task")
st.markdown( 
    """
    ### <span style="color:green"> This are list of work in progress and plan for improvement </span>
    """
,unsafe_allow_html=True)


genre = st.radio(
    "# Backend Development",
    ('RCI Development', 'Experiment with different Prompt', 
     'Enable Memory for different RAG','Develop correction mechanism', 'Re-train the model using QLora',
     'Experiment RAG with llama-index', 'Create API that enable model to connect to SpecXplorer, FeatureXplorer', 'Discuss with Andreas Klein for Elastic Vector Database indexing' ))

genre = st.radio(
    "# Frontend Development",
    ('Clean the dataset', 'Learn how to correctly split the data','Advise team to use standard FAQ template','Update the documentation','Assign Specialist to review individual FAQ'))

genre = st.radio(
    "# Deplyoment Method",
    ('Discover the best approach to deploy the Wiki', 'Experiment to use Docker'))

genre = st.radio(
    "Resources",
    ('Backend Developer - that can help to to improve the backend architecture and experiment with different retrieval method.',
     'Frontend Developer - that can help to focus on UI development', 'Intern to help with dataset','Training Facility'))

# if genre == 'Comedy':
#     st.write('You selected comedy.')
# else:
#     st.write("You didn\'t select comedy.")