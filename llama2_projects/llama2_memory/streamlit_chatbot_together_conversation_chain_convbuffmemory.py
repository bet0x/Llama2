import streamlit as st

import pandas as pd
import numpy as np
import random

from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate, LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import ElasticVectorSearch
from streamlit_chat import message
import asyncio
import textwrap

import together
from togetherllm import TogetherLLM
from time import sleep

from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE

from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

st.set_page_config(page_title='🧠MemoryBot🤖', layout='wide')

st.title('🎈 Llama 2 Chatbot | Entity Memory')

llm = TogetherLLM(
model= "togethercomputer/llama-2-7b-chat",
temperature=0,
max_tokens=512
)

# Set Streamlit page configuration
# Initialize session states
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []

with st.sidebar:
    st.title("X-Chat")
    st.header("Settings")
    add_replicate_api = st.text_input("Enter your password here", type='password')

    if not (add_replicate_api.startswith('jl')):
        st.warning('Please enter your credentials', icon='⚠️')
    else:
        st.success('Proceed to entering your prompt message!', icon='👉')

if 'entity_memory' not in st.session_state:
    st.session_state.entity_memory = ConversationBufferMemory()

Conversation = ConversationChain(
    llm=llm,
    memory = st.session_state.entity_memory,
)

if ask := st.chat_input(st.session_state["input"],  disabled=not add_replicate_api):
    st.session_state.messages.append({"role": "user", "content": ask})
    with st.chat_message("user"):
        st.write(ask)


def generate_response(prompt):
    if prompt:
        output = Conversation.run(input=ask) 
        return output

    
# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]
    

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

#Clear the chat messages
def clear_chat_history():
   st.session_state.messages=[{"role": "assistant","content": "How may i assist you today ?"}]

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking"):
            response = generate_response(ask)
            
            st.session_state.past.append(ask)
            st.session_state.generated.append(response)
            
            placeholder = st.empty()
            full_response=''
            for item in response:
                full_response+=item
                sleep(0.01)
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
            with st.expander("Conversation"):
                for i in range(len(st.session_state['generated'])-1, -1, -1):
                    st.info(st.session_state["past"][i],icon="🧐")
                    st.success(st.session_state["generated"][i], icon="🤖")
                    
    message = {"role":"assistant", "content": full_response}
    st.session_state.messages.append(message)
