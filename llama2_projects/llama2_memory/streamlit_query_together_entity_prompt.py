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
import asyncio
import textwrap

import together
from togetherllm import TogetherLLM
from time import sleep

from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE

template = """[INST] <<SYS>>
You are helpful assistant, you always only answer for the assistant then you stop, read the chat history to get the context
{chat_history}

Question: {user_input}
<</SYS>>

[/INST]"""

print(template)

st.set_page_config(page_title='🧠MemoryBot🤖', layout='wide')

st.title('🎈 Llama 2 Chatbot')

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


# Define function to get user input
def get_text():
    """
    Get the user input text.

    Returns:
        (str): The text entered by the user
    """
    input_text = st.text_input("You: ", st.session_state["input"], key="input",
                            placeholder="Your AI assistant here! Ask me anything ...", 
                            label_visibility='hidden')
    return input_text

with st.sidebar:
    st.title("X-Chat")
    st.header("Settings")
    add_replicate_api = st.text_input("Enter your password here", type='password')

    if not (add_replicate_api.startswith('jl')):
        st.warning('Please enter your credentials', icon='⚠️')
    else:
        st.success('Proceed to entering your prompt message!', icon='👉')

# Create a ConversationEntityMemory object if not already created
if 'entity_memory' not in st.session_state:
    st.session_state.entity_memory = ConversationEntityMemory(llm=llm, k=3 )
    
llm_chain=LLMChain(prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE, memory=st.session_state.entity_memory, llm=llm ,verbose=True)
        
user_input = get_text()

if user_input:     
    output = llm_chain.run(user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)
        
with st.expander("Conversation"):
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        st.info(st.session_state["past"][i],icon="🧐")
        st.success(st.session_state["generated"][i], icon="🤖")
