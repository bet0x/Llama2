from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.vectorstores import ElasticVectorSearch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain

from togetherllm import TogetherLLM
import logging

import streamlit as st
from st_pages import Page, Section, show_pages, add_page_title
from st_pages import show_pages_from_config
from langchain.prompts import PromptTemplate
import warnings

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

st.set_page_config(
    page_title="Wikipedia",
     page_icon="https://api.dicebear.com/5.x/bottts-neutral/svg?seed=gptLAb"#,
)

add_page_title()

template = """[INST] <<SYS>>
Your name is Kelly, you are a helpful, respectful and honest assistant. Always answer as helpfully as possible using the context text provided.
You answer should only answer the question once and not have any text after the answer is done.\n\nIf a question does not make any sense, or is not factually
coherent, explain why instead of answering something not correct. If you don't know the answer, just say you don't know and submit the request to hotline@xfab.com for further assistance.\n
<</SYS>>
CONTEXT:/n/n {context}/n
{chat_history}
Question: {question}
[/INST]"""

# template = """Use the following pieces of context to answer the question at the end. 
# If you don't know the answer, just say that you don't know, don't try to make up an answer. 
# Use three sentences maximum and keep the answer as concise as possible. 
# Always say "thanks for asking!" at the end of the answer. 
# {context}
# Question: {question}
# Helpful Answer:"""

print(template)

prompt = PromptTemplate(input_variables=["chat_history","question","context"], template=template)
memory = ConversationBufferMemory(input_key="question",memory_key="chat_history",)

st.title("Load Chain Retrieval QA")
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

def semantic_search(docsearch,query):
    docs=docsearch.similarity_search(query)
    return docs

logging.basicConfig()
logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)

vectorstore = load_db()
chain=load_qa_chain(llm, chain_type="stuff",memory=memory, prompt=prompt)

if question:
    with st.expander("Return result"):
        semantic_result = semantic_search(vectorstore,question)
        chain_input={
            "input_documents":semantic_result,
            "question":question
        }
        llm_result = chain(chain_input,return_only_outputs=True)
        res = llm_result['output_text']
        st.info(res)

        #res = qa_chain({"query": question})
        #st.info(res['result'])
        #st.info(res['source_documents'][0])

   