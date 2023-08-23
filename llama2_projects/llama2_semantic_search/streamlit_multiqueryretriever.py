from langchain.retrievers.multi_query import MultiQueryRetriever
from togetherllm import TogetherLLM
from langchain.vectorstores import ElasticVectorSearch
from langchain.embeddings import HuggingFaceEmbeddings

import together
import logging

import streamlit as st
from st_pages import Page, Section, show_pages, add_page_title
from st_pages import show_pages_from_config

st.title("Multiquery Retriever")
prompt=st.text_input("Enter your question here")

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

retriever_from_llm = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(search_kwargs={'k': 1}),llm=llm)
#unique_docs = retriever_from_llm.get_relevant_documents(query="What is flatpv ?")
#print(unique_docs)

if prompt:
    with st.expander("Return result"):
        answer = retriever_from_llm.get_relevant_documents(query=prompt)
        for i in range(len(answer)):
            st.info(answer[i].page_content)