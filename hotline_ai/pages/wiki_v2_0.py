import streamlit as st
from st_pages import Page, Section, show_pages, add_page_title
from st_pages import show_pages_from_config
from langchain import ElasticVectorSearch
from langchain.embeddings import HuggingFaceEmbeddings
from streamlit_chat import message

st.set_page_config(
    page_title="Wikipedia",
     page_icon="https://api.dicebear.com/5.x/bottts-neutral/svg?seed=gptLAb"#,
)

#embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

CERT_PATH = "D:\elasticsearch-8.4.2\config\certs\http_ca.crt"
ELASTIC_PASSWORD = "Eldernangkai92"

elasticsearch_url = f"https://elastic:Eldernangkai92@localhost:9200"

add_page_title()

st.markdown("""### <span style="color:green"> In this revise architecture, we're using Vector Database and Semantic approach as our retreival system.</span>""",unsafe_allow_html=True)
st.markdown("""
            * Notice that `return` response is a chunk data with relevant answer.
            * We need to use `chunk` data because Llama2 can accept maximum input token data up to 4026. 1 Token is equivalent to 4 English words.
            """)
st.title("Wiki 2.0 Encoder")
prompt=st.text_input("Enter your question here")

def load_db():
    db= ElasticVectorSearch(
        elasticsearch_url=elasticsearch_url,
        index_name="new_wikidb_v1",
        ssl_verify={
            "verify_certs": True,
            "basic_auth": ("elastic", ELASTIC_PASSWORD),
            "ca_certs": CERT_PATH,
        },
        embedding=embeddings
    )

    return db

def search(query):
    db = load_db()

    docs = db.similarity_search(query)
    #x = (docs[0].page_content)
    
    return docs

if prompt:
   data = []
   with st.expander("Return result"):
      #data.append(search(prompt))
      x = search(prompt)
      for i in range(len(x)):
          st.info(x[i].page_content)

with st.expander("Read More"):
    col1, col2= st.columns(2)

    with col1:
        st.header("Pros")
        st.markdown(""" 
                    ### Improved Relevance
                    * The retriever can understand synonyms, acronyms, and variations of terms,
                    reducing the likelihood of missing relevant content due to differences in terminology

                    ### Support chain of question
                    * The retriever is based on BERT model. Therefore, it can support more complex and natural
                    human prompt and return releveant content due to its capability to perform text classification.
                    * User able to input their queries in a more conversational and natural manner.
                    * Retriever will return all relevant answer based on user input.

                    ### Support Vector Data
                    * Vector Data (Embedded) is quite popular now adays due to its speed and scability.
                    * With arising number of ticets, therefore it's recommended to convert `dataset` as vector data
                    because it support semantic search as part of the retrieval system. 
                    * In this project, i'm using vector database (Elastic Vector) as a knowledge base.
                    """)
        st.code("""
            Example of Question:
            * What is flatpv ?  and how to bias fifth terminal of HW wafer ?
        """)   
    
    with col2:
        st.header("Cons")
        st.markdown("""
            ### Non Generative model
            * Unable to generate creative answer based on return response.
            * Return result usually in ranked top response with relate to user input.
            """)
        
with st.expander("Sample Output"):
    st.image("./images/wiki_2_0_image.jpg", caption="Fig 1: Semantic result")

           
      # print(x)
      # print("\n")
   


      # for i in range(0,):
      #    data.append(get)
      
      # for i in range(0,4):
      #    res = data[i]
      #    st.info(res)

          

# if 'history' not in st.session_state:
#     st.session_state['history'] = []

# # Output
# if 'generated' not in st.session_state:
#     st.session_state['generated'] = ["Hello ! Ask me about anything " + "😊"]
    
# # User Input
# if 'past' not in st.session_state:
#     st.session_state['past'] = ["Hey ! 👋"]

# #container for the chat history
# response_container = st.container()

# #container for the user's text input
# container = st.container()

# with container:
#     with st.form(key='my_form', clear_on_submit=True):
        
#         user_input = st.text_input("Query:", placeholder="Search your data here :", key='input')
#         submit_button = st.form_submit_button(label='Send')
        
#     if submit_button and user_input:
#         output = search(user_input)
        
#         st.session_state['past'].append(user_input)
#         st.session_state['generated'].append(output)

# if st.session_state['generated']:
#     with response_container:
#         for i in range(len(st.session_state['generated'])):
#             message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
#             message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")













# tab1, tab2, tab3 = st.tabs(["Cat", "Dog", "Owl"])

# with tab1:
#    st.header("A cat")
#    st.image("https://static.streamlit.io/examples/cat.jpg", width=200)

# with tab2:
#    st.header("A dog")
#    st.image("https://static.streamlit.io/examples/dog.jpg", width=200)

# with tab3:
#    st.header("An owl")
#    st.image("https://static.streamlit.io/examples/owl.jpg", width=200)