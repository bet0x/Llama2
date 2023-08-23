from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.vectorstores import ElasticVectorSearch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory

from togetherllm import TogetherLLM
import logging

import streamlit as st
from st_pages import Page, Section, show_pages, add_page_title
from st_pages import show_pages_from_config
from langchain.prompts import PromptTemplate

template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use three sentences maximum and keep the answer as concise as possible. 
Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

st.title("Basic Retrieval QA")
question=st.text_input("Enter your question here")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

llm = TogetherLLM(
    model= "togethercomputer/llama-2-7b-chat",
    temperature=0,
    max_tokens=512
)

def load_db():
    CERT_FINGERPRINT = "7e73d3cf8918662a27be6ac5f493bf55bd8af2a95338b9b8c49384650c59db08"
    CERT_PATH = "D:\elasticsearch-8.4.2\config\certs\http_ca.crt"
    ELASTIC_PASSWORD = "Eldernangkai92"
    elasticsearch_url = f"https://elastic:Eldernangkai92@localhost:9200"

    db= ElasticVectorSearch(
        elasticsearch_url=elasticsearch_url,
        index_name="new_wikidb_v1",
        ssl_verify={
            "verify_certs": True,
            "basic_auth": ("elastic", ELASTIC_PASSWORD),
            "ssl_assert_fingerprint" :  CERT_FINGERPRINT # You can use fingerprint also
            #"ca_certs": CERT_PATH, # You can Certificate path too
        },
        embedding=embeddings
    )
    
    return db

logging.basicConfig()
logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)

vectorstore = load_db()
qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectorstore.as_retriever(search_kwargs={'k': 1}), chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}, return_source_documents=True)

if question:
    with st.expander("Return result"):
        res = qa_chain({"query": question})
        st.info(res['result'])
        st.info(res['source_documents'][0])

   