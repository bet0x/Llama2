import streamlit as st
from st_pages import Page, Section, show_pages, add_page_title
from st_pages import show_pages_from_config
import pandas as pd
import numpy as np
import random

from langchain import PromptTemplate, LLMChain
from langchain.memory import ConversationBufferMemory
from streamlit_chat import message

import together
from togetherllm import TogetherLLM

st.set_page_config(
    page_title="Wikipedia",
     page_icon="https://api.dicebear.com/5.x/bottts-neutral/svg?seed=gptLAb"#,
)

add_page_title()

custom_prompt_template = """[INST] <<SYS>>
You are helpful assistant, you always only answer for the assistant then you stop, read the chat history to get the context
{chat_history}

Question: {user_input}
<</SYS>>

[/INST]"""

print(custom_prompt_template)

# custom_prompt_template = """[INST] <<SYS>>
# Your name is Kelly.
# Kelly is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.
# Kelly is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.
# Overall, Kelly is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.
# <</SYS>>
# {chat_history}

# Human: {user_input}
# Assistant:
# [/INST]
# """

#print(custom_prompt_template)

#@st.cache_resource()

@st.cache_data(experimental_allow_widgets=True)  # 👈 Set the parameter
def chat_model():
    llm = TogetherLLM(
        model= "togethercomputer/llama-2-7b-chat",
        temperature=0,
        max_tokens=512
    )
    return llm

memory = ConversationBufferMemory(input_key="user_input", memory_key="chat_history",)
main_prompt = PromptTemplate(input_variables=['chat_history','user_input'], template=custom_prompt_template)
llm = chat_model()

#LLM_Chain=LLMChain(prompt=prompt, memory=memory, llm=llm)

LLM_Chain = LLMChain(
    llm=llm,
    prompt=main_prompt,
    verbose=False,
    memory=memory,
    llm_kwargs={"max_length": 4096}
)

with st.sidebar:
    st.title("X-Chat")
    st.header("Settings")
    add_replicate_api = st.text_input("Enter your password here", type='password')
    # os.environ["TOGETHER_API_KEY"] = "4ed1cb4bc5e717fef94c588dd40c5617616f96832e541351195d3fb983ee6cb5"

    if not (add_replicate_api.startswith('jl')):
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
   st.session_state.messages=[{"role": "assistant","content": "How may i assist you today sir ?"}]

for message in st.session_state.messages:
   with st.chat_message(message['role']):
       st.write(message['content'])

#Clear the chat messages
def clear_chat_history():
   st.session_state.messages=[{"role": "assistant","content": "How may i assist you today ?"}]

#Clear the chat messages
st.sidebar.button('Clear Chat History', on_click=clear_chat_history())


# User provided prompt
if prompt := st.chat_input(disabled= not add_replicate_api):
    st.session_state.messages.append({"role":"user", "content":prompt})  
    with st.chat_message("user"):
        st.write(prompt)   

if st.session_state.messages[-1]["role"] != "assistant":
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking"):
            #response = LLM_Chain.run(prompt)
            response = LLM_Chain.predict(user_input=prompt)
            placeholder = st.empty()
            full_response=''
            for item in response:
                full_response+=item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role":"assistant", "content": full_response}
    st.session_state.messages.append(message)
