from langchain import ElasticVectorSearch
from langchain.document_loaders import PyPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.chains.question_answering import load_qa_chain
from elasticsearch import Elasticsearch
from langchain.chains import RetrievalQA

import streamlit as st
from streamlit_chat import message

PATH = r"D:/AI_CTS/Llama2/llama2_projects/llama2_pdf_chatbot_faiss_windows/"

DATA_PATH = PATH + 'data/'
DB_FAISS_PATH = PATH + 'vectorstore/db_faiss'

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

CERT_PATH = "D:\elasticsearch-8.4.2\config\certs\http_ca.crt"
ELASTIC_PASSWORD = "Eldernangkai92"

elasticsearch_url = f"https://elastic:Eldernangkai92@localhost:9200"

# Using object notation
add_selectbox = st.sidebar.selectbox(
    "How would you like to be contacted?",
    ("Email", "Home phone", "Mobile phone")
)

# Using "with" notation
with st.sidebar:
    add_radio = st.radio(
        "Choose a shipping method",
        ("Standard (5-15 days)", "Express (2-5 days)")
    )

st.title("Hotline Virtual Assistant👩‍💼")
st.markdown("<h3 style='text-align: center; color: white;'>Built by <a href='https://github.com/AIAnytime'>X-Fab with ❤️ </a></h3>", unsafe_allow_html=True)


def load_db():
    db= ElasticVectorSearch(
        elasticsearch_url=elasticsearch_url,
        index_name="elastic_vector",
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
    x = (docs[0].page_content)
    print(x)

    return x
 
if 'history' not in st.session_state:
    st.session_state['history'] = []

# Output
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello ! Ask me about anything " + "😊"]
    
# User Input
if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey ! 👋"]

#container for the chat history
response_container = st.container()

#container for the user's text input
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        
        user_input = st.text_input("Query:", placeholder="Search your data here :", key='input')
        submit_button = st.form_submit_button(label='Send')
        
    if submit_button and user_input:
        output = search(user_input)
        
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
            message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
