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
    "Task",
    ('RCI Development', 'Experiment with different Prompt', 'Clean the dataset', 
     'Enable Memory for different RAG','Develop correction mechanism', 'Re-train the model using QLora',
    'Learn how to correctly split the data','Experiment RAG with llama-index', 'Advise team to correctly format their dataset',
    'Create API that enable model to conenct to SpecXplorer, FeatureXplorer'))

# if genre == 'Comedy':
#     st.write('You selected comedy.')
# else:
#     st.write("You didn\'t select comedy.")