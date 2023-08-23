import streamlit as st
import pandas as pd
from st_pages import Page, Section, show_pages, add_page_title
from st_pages import show_pages_from_config
from elastic_api import *
from termcolor import colored

st.set_page_config(
    page_title="Wikipedia",
     page_icon="https://api.dicebear.com/5.x/bottts-neutral/svg?seed=gptLAb"#,
)

add_page_title()

def highlight_result(result, keyword):
    highlighted_result = result.replace(keyword, colored(keyword, 'red', attrs=['bold']))
    return highlighted_result

st.markdown("""
### <span style="color:green"> In this early architecture, we're using Elastic Search pattern matching approach </span>
""",unsafe_allow_html=True)

es = elastic()
es.connect()

st.title("Wiki 1.0 Elastic Search")
prompt=st.text_input("Enter your question here")
read = es.search(prompt)

res = read['hits']['hits']
print(read)

length = len(res)

if prompt:
    data = []
    with st.expander(" Return result"):
        for i in range(0,length):
            data.append([res[i]['_source']['Question'],
                         res[i]['_source']['Answer'],
                         res[i]['_source']['Topic']])
            
        for i in range(0,length):
            # result_fields = " ".join(map(str, data[i]))  # Combine fields into a single string
            # highlighted_result = highlight_result(result_fields, prompt)
            # st.markdown(str(highlighted_result))
            # print(highlighted_result)

            res = highlight_result(str(data[i]), prompt)
            st.info(res)
            print(res)

with st.expander("Read More"):
    col1, col2= st.columns(2)

    with col1:
        st.header("Pros")
        st.markdown(""" 
                    ### Advantageous when it comes to precision
                    * The retriever ensures that return content exactly matched the desired input.
                    * The retriever searches for exact matches of the provided query, character. It looks
                    for the sequence of characters.

                    ### Coherent understanding of natural human query.
                    * The retreiver able to understand simple natural prompt and return an associated content.
                    """)
    
    with col2:
        st.header("Cons")
        st.markdown("""
                    ### Didn't support chain of questions.
                    * Elastic retriever is not build to accept chain of question which lead to nonsene and no result.
                    * The retriever unable to support 2 separate question in one prompt.
                    """)
        st.code("""
                    Example of Question:
                    * What is flatpv ?  and how to bias fifth terminal of HW wafer ?
                """)   

         
        st.markdown("""
            ### Limited scability of database.
            * The dataset is stored as a metadata (Json) in elastic database.
            * This may work only with elastic but not scalable to other database platform as well when it comes to data migration.
            * Therefore it's not suitable to be used as a general knowledge base.
            """)   
        
# col1, col2, col3 = st.columns(3)

# with col1:
#    st.header("A cat")
#    st.image("https://static.streamlit.io/examples/cat.jpg")

# with col2:
#    st.header("A dog")
#    st.image("https://static.streamlit.io/examples/dog.jpg")

# with col3:
#    st.header("An owl")
#    st.image("https://static.streamlit.io/examples/owl.jpg")