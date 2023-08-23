import streamlit as st
from st_pages import Page, Section, show_pages, add_page_title
from st_pages import show_pages_from_config
import pandas as pd
import numpy as np
import random

from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate, LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import ElasticVectorSearch
from streamlit_chat import message

import together
from togetherllm import TogetherLLM

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

st.set_page_config(
    page_title="Wikipedia",
     page_icon="https://api.dicebear.com/5.x/bottts-neutral/svg?seed=gptLAb"#,
)

add_page_title()

custom_prompt_template = """[INST] <<SYS>>
Your name is Kelly, you are a helpful, respectful and honest assistant. Always answer as helpfully as possible using the context text provided.
You answer should only answer the question once and not have any text after the answer is done.\n\nIf a question does not make any sense, or is not factually
coherent, explain why instead of answering something not correct. If you don't know the answer, just say you don't know and submit the request to hotline@xfab.com for further assistance.\n
<</SYS>>
CONTEXT:/n/n {context}/n
{chat_history}
Question: {question}
[/INST]"""

print(custom_prompt_template)

# st.title("Wiki 3.0 Decoder Encoder")
# prompt=st.text_input("Enter your question here")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
memory = ConversationBufferMemory(input_key='question', memory_key='chat_history', return_messages=True)

# Top_p: This is like setting a rule that the AI can only choose from the best possible options. If you set top_p to 0.1, it's like telling the AI, "You can only pick from the top 10% of your 'best guesses'."
# Top_k: This one is similar to top_p but with a fixed number. If top_k is set to 50, it's like telling the AI, "You have 50 guesses. Choose the best one."
llm = TogetherLLM(
    model= "togethercomputer/llama-2-7b-chat",
    temperature=0,
    max_tokens=512
)

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(input_variables=['chat_history','context', 'question'],
                            template=custom_prompt_template)
    return prompt

def conversationalretrieval_qa_chain(llm, prompt, db, memory):
    chain_type_kwargs = {"prompt": prompt}
    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                     chain_type= 'stuff',
                                                     retriever=db.as_retriever(search_kwargs={'k': 3}),
                                                     verbose=True,
                                                     memory=memory,
                                                     combine_docs_chain_kwargs=chain_type_kwargs
                                                     )
    return qa_chain


def load_db():
    CERT_FINGERPRINT = "7e73d3cf8918662a27be6ac5f493bf55bd8af2a95338b9b8c49384650c59db08"
    CERT_PATH = "D:\elasticsearch-8.4.2\config\certs\http_ca.crt"
    ELASTIC_PASSWORD = "Eldernangkai92"
    elasticsearch_url = f"https://elastic:Eldernangkai92@localhost:9200"

    db= ElasticVectorSearch(
        elasticsearch_url=elasticsearch_url,
        index_name="new_wikidb",
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

#QA Model Function
def qa_bot(ask):

    db = load_db()
    prompt = set_custom_prompt()
    qa = conversationalretrieval_qa_chain(llm, prompt, db, memory)
    
    result = qa({"question": ask})
    res = result['answer']

    #res = llm(ask)
    #print(res)
    #return res

    # llm_chain = LLMChain(
    # llm=llm,
    # prompt=prompt,
    # verbose=True,
    # memory=memory,)

    # res = llm_chain.predict(user_input=ask)
    #print(res)

    return res

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

def revise():
    st.session_state.editmeplease

with container:
    with st.form(key='my_form', clear_on_submit=True):
        
        user_input = st.text_input("Query:", placeholder="Search your data here :", key='input')
        submit_button = st.form_submit_button(label='Send')
        
    if submit_button and user_input:
        output = qa_bot(user_input)
        
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
            message(st.session_state["generated"][i], key=str(i), avatar_style="avataaars")

with st.expander('Chat History'):
        st.info(memory.buffer)

st.markdown("<h6 style='text-align: right; color: white;'>Built by <a href='https://github.com/AIAnytime'>X-Fab with ❤️ </a></h3>", unsafe_allow_html=True)
