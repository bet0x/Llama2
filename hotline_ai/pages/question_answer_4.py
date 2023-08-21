import streamlit as st
from st_pages import Page, Section, show_pages, add_page_title
from st_pages import show_pages_from_config

add_page_title()

st.markdown("## Accuracy Improvement Method")
st.markdown( 
    """
    * To date, i had identify few method that can be used to improve the accuracy of the `response` from the system. Of course, this will be under
    continous study and experiment.
   
    """
)

st.image("./images/q_a_4.jpg", caption="Fig 2: Hardware Requirements")

st.markdown("### Following are accuracy improvement method")

with st.expander("Quality Data Preprocessing"):
    st.markdown("""
    * Clean and preprocess dataset thoroughly. This include handling missing values, use proper sentence and wording, avoid abbreviations without context and performing text normalization (lowercase, stemming and etc).    
    """)

with st.expander("Use correct prompt for dataset provide details answer"):
    st.markdown("""
    * Avoid using refer to ticket with no details context provided.  
    * To write question and answer in full english sentence and in complete paragraph.
    * Justify the abbreviation.
    * Use prompt format as follow that should include `Can you tell me`  for question.
    * Use `Certainly, I'd be happy to help you with it` and repeat the question as part of the prompt
    `To answer your question regarding on how 
    are the MLM layers grouped in each reticle`
    """)
    st.code("""
    Question
    Can you tell me, How are the MLM layers grouped in each reticle ? 

    Answer
    Certainly, I'd be happy to help you with it. To answer your question regarding on how 
    are the MLM layers grouped in each reticle, I suggest you to visit  AX device page, you can 
    go to MGO tab on right side to view. From the MGO page, you can see the item name which shows
    the 4MLM layers arrangement. Normally we don't disclose the barcode of item number to customer. 
    You can extract the info into the excel format at the top.
    
    """)

with st.expander("Data Augmentation"):
    st.markdown("""
    * Augment the dataset by generating new example from it with slight modifications such as synonyms, paraphrases, or word substitutions. This can help the model to a broader range of language variations.    
    """)

with st.expander("Fine-Tune Bert Embedded Model"):
    st.markdown("""
    * Fine-tune BERT model on sentence similarity based on custom dataset improve the adaptability and familiarity of the model to that specific token. Thus provide a good return search result and experiment with different hyperparameters like batch size, learning rate, dropout rate and optimizer.    
    """)

with st.expander("Use Top Massive Text Embedding Models"):
    st.markdown("""
    * To use task-specific pre-trained embedded model. These model may have already been tuned for similar task. For example, information retrieval, text similarity and etc.
    * You can find list of top model at [MTEB]("https://huggingface.co/spaces/mteb/leaderboard). Higher benchmark is better for classification.
    """)

with st.expander("Explore different method of Retrieval"):
    st.markdown("""
    * Currently, i'm still experimenting and developing RCI (Recursive, Criticsm and Implementation) method that allow the LLM model
    to critise their own answer and correct themselves based on the original question.
 
    """)


