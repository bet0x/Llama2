import streamlit as st
from st_pages import Page, Section, show_pages, add_page_title
from st_pages import show_pages_from_config

add_page_title()

st.markdown("""### <span style="color:red">The problem that we have with LLM, is that this model didn't have an access to outside world knowledge. The only knowledge contained within them is knowledge they learn from training, which can be very limiting. </span>""",unsafe_allow_html=True)

st.markdown("## Challenges with Fine-Tuning llama2 model")

st.markdown( 
    """
    ### Time Consuming 
    * Fine-tuning a model requires a significant time investment. This includes training and optimizing time
      for the model, in addition to determining the best practices and techniques for your approach

    ### ‍Infrastructure overhead
    * X-Fab didn't have any training infrasctructure to train the model with large dataset.
    * Training the model often require high spec of GPU hardware as it's computationally extensive.
    * There is a 3-rd party GPU resources such as `Google Colab` that support  model training. But due to 
      security data reason, we choose not to train Llama2 model using this facility.
    """
)

st.image("./images/q_a_1.jpg", caption="Fig 1: Fine-tuning challenges")

st.markdown( 
    """
    ## Hardware Requirements 
    * Below is the suggested hardware requirement to train this model.
    """
)

st.image("./images/q_a_2.jpg", caption="Fig 2: Hardware Requirements")
