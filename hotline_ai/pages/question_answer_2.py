import streamlit as st
from st_pages import Page, Section, show_pages, add_page_title
from st_pages import show_pages_from_config

add_page_title()

st.markdown("## Alternative to model Fine-Tuning")
st.markdown( 
    """
    ### Using Retrieval Argumented (RAG)
    * The idea behind RAG, basically we give LLM access to the outside world without re-training the model.
    * The way we're going to do it, at least in this example is by using method of searching with natural language which is ideal when it comes
    to our LLM because llama2 works with Natural language. 
    * So basically we interact with LLM using natural langugae and then we search with natural language that allow us to ask a question and then we'll get relevant information about
    that question from the `knowledge base` and we get to feed that relevant information plus initial our initial question into llama 2 giving it access.

    """
)

st.image("./images/llm_db.jpg", caption="Fig 1: Fine-tuning challenges")
st.markdown( 
    """
    * For this we'll need to use embedding model. Embedding model is how we build this retrieval system
    * Basically is how we translate human readable text into machine readable vector.
    * We need machine readable vector in order to perform a search based on `semantic` meaning rather than like `traditional` search which is based on `text` and `keywords`

    """
)

st.markdown( 
    """
    ## System Architecture Wiki 2.0 | Encoder Only
    * Here we implement the retrieveal system based on embedding and `semantic` search.
    * The documents (*pdf, *csv, *json) contents will be spliitted and chunks due to the limitations of llama2 token
    that only able to accept maximum input token of `4026`. 1 Token is equivalent to `4` english words.
    * The chunk data then will be converted to vector data (floating point) by embedded model and store into `Knowledge Based`.
    * Most poupular Vector Database out there are :
        * pineceone
        * chromadb
        * FAISS
        * elastic search
        * etc
    * For this architecture, we're using `Elastic Search Vector Database` as our Knowledge based.
    """
)

st.image("./images/wiki_2_0_Encoder.jpg", caption="Fig 2: Hardware Requirements")

st.markdown( 
    """
    * User will prompt a question, and this question will be converted into Vector data by the embedded model.
    * This question will be used as part of the retrieval. Here, we're using `Semantic Search` to return most relevant or top rank data from the Vector Database.
    * This response then will be returned to the user.
    """
)

col1, col2= st.columns(2)

with col1:
   st.header("Pros")
   st.markdown(""" 
               ### Improved Relevance
               * Semantic search understands the context and intent of user queries, leading to more accurate and relevant search results.
               
               ### Natural Language Processing
               * Enable users to input queries in a more conversational and natural manner.
               
               ### Conceptual Understanding
               * Semantic search can understand synonyms, acronyms, and variations of terms, reducing the likelihood of missing relevant content due to differences in terminology
               
               ### Speed
               * It's relatively fast.
               """)
   

with col2:
   st.header("Cons")
   st.markdown("""
               ### Initial Setup
               * Creating a semantic search system involves training models, setting up ontologies or taxonomies, and mapping relationships between concepts
               
               ### Accuracy Challenges
               * While semantic search aims to improve relevance, it might still face challenges in correctly interpreting the user's intent, especially in cases where the query is highly context-dependent or ambiguous
               
               ### Maintenance and Updates
               * As languages evolve and new concepts emerge, semantic search systems need to be regularly updated to stay effective and relevant
               
               ### Linguistic Variation
               * Different users might use different phrasing to refer to the same concept. Semantic search systems need to handle this linguistic variation effectively to provide accurate results.
               """)
