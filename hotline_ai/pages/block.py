import streamlit as st
from st_pages import Page, Section, show_pages, add_page_title
from st_pages import show_pages_from_config

add_page_title()

st.title("New Wiki Backend Architecture")
st.markdown( 
    """
    ### Blocks Descriptions
    * The overall system architecture is structured into 3 distinct block: :green[Encoder], :green[Decoder] and :green[Knowledge Base]. 
    Each block serves a specific function within the AI model.

    """
)

st.image("./images/new_block.jpg", caption="Fig 1: New Backend Block Description")

st.title("Model Selection and Evaluation")
st.markdown( 
    """
    ## llama 2
    * Second generation open source large language model (LLM) from Meta. It can be used to build
    chatbots like ChatGPT, Google Bard and Claude.
    * This mode is free to use, even for commercial applications, under a new open-source licence
    * Visit [Meta](https://ai.meta.com/llama/) to check their official documentation.

    """
)

st.image("./images/llama2.jpg", caption="Fig 1: Llama 2 evaluation")


st.markdown( 
    """
    ## BERT
    * Is open source machine learning model for natural language processing (NLP).
    * BERT is designed to help computers understand the meaning of ambiguous language in text by using
    surrounding text to establish right context.
    """
)

st.image("./images/bert.jpg", caption="Fig 2: BERT evaluation")