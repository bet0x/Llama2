import streamlit as st
from st_pages import Page, Section, show_pages, add_page_title
from st_pages import show_pages_from_config
import pandas as pd
import numpy as np
import random

from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate, LLMChain
from langchain.memory import ConversationBufferMemory

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

memory = ConversationBufferMemory(input_key="user_input",memory_key="chat_history",)

llm = TogetherLLM(
    model= "togethercomputer/llama-2-7b-chat",
    temperature=0,
    max_tokens=512
)

with st.sidebar:
    st.title("X-Chat")
    st.header("Settings")
    add_replicate_api = st.text_input("Enter your api key here", type='password')
    # os.environ["TOGETHER_API_KEY"] = "4ed1cb4bc5e717fef94c588dd40c5617616f96832e541351195d3fb983ee6cb5"

    if not (add_replicate_api.startswith('4e')):
        st.warning('Please enter your credentials', icon='⚠️')
    else:
        st.success('Proceed to entering your prompt message!', icon='👉')

    st.subheader("Models and Parameters")
    select_model = st.selectbox("Choose AI model", ['Llama 27b', 'Llama 2 13b','Llama 2 70b'], key='select_model')

    if select_model == 'Llama 2 7b':
        model = "togethercomputer/llama-2-7b-chat"
    
    elif select_model == "Llama 2 13 b":
        model = "togethercomputer/llama-2-13b-chat"

    elif select_model == "Llama 2 70b":
        model = "togethercomputer/llama-2-7b-chat"
    

if "messages" not in st.session_state.keys():
    st.session_state.messages=[{"role": "assistant","content": "How may i assist you today ?"}]

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.write(message['content'])

# Clear the chat messages
def clear_chat_history():
    st.session_state.messages=[{"role": "assistant","content": "How may i assist you today ?"}]

st.sidebar.button('Clear Chat History', on_click=clear_chat_history())

# Create a function to generate llama 2 response
def generate_response(prompt_input):
    #default_system_prompt="You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as a 'Assistant'. {user_input} {chat_history}"
    
    default_system_prompt = """[INST] <<SYS>>
    You are helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as a 'Assistant'.
    you always only answer for the assistant then you stop, read the chat history to get the context.
    <</SYS>>
    {chat_history}
    Question: {user_input}

    [/INST]"""

    # default_system_prompt = """[INST] <<SYS>>
    # You are helpful assistant, you always only answer for the assistant then you stop, read the chat history to get the context
    # {chat_history}

    # Question: {user_input}
    # <</SYS>>

    # [/INST]"""

    for data in st.session_state.messages:
        print("Data :", data)
        if data['role'] == "user":
            default_system_prompt+="User: " + data['content'] + "\n\n"
        else:
            default_system_prompt+="Assistant, " + data['content'] + "\n\n"

    print(default_system_prompt)
    prompt = PromptTemplate(input_variables=["chat_history", "user_input"], template=default_system_prompt)
    
    llm_chain = LLMChain(prompt=prompt,memory=memory, llm=llm)
    res = llm_chain.run(prompt_input)
    print(res)
    return res

# User provided prompt
if prompt := st.chat_input(disabled= not add_replicate_api):
    st.session_state.messages.append({"role":"user", "content":prompt})  
    with st.chat_message("user"):
        st.write(prompt)      

# Generate new response if last message is not from the assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking"):
            response = generate_response(prompt)
            placeholder = st.empty()
            full_response=''
            for item in response:
                full_response+=item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role":"assistant", "content": full_response}
    st.session_state.messages.append(message)



