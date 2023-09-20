import streamlit as st
from st_pages import Page, Section, show_pages, add_page_title
from st_pages import show_pages_from_config
import pandas as pd
import numpy as np

from langchain import PromptTemplate, LLMChain
from langchain.memory import ConversationBufferMemory
from streamlit_chat import message

from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

from time import sleep

st.set_page_config(
    page_title="Chatbot Playground",
     page_icon="https://api.dicebear.com/5.x/bottts-neutral/svg?seed=gptLAb"#,
)

#add_page_title()

# GGUF
MODEL_PATH = r"/home/jlukas/Desktop/My_Project/AI_CTS/Llama2_Quantized/7B_GGUF/llama-2-7b-chat.Q4_K_M.gguf"

custom_prompt_template = """[INST] <<SYS>>
You are helpful assistant, you always only answer for the assistant then you stop, read the chat history to get the context
{chat_history}

Question: {user_input}
<</SYS>>

[/INST]"""

print(custom_prompt_template)

@st.cache_data(experimental_allow_widgets=True)  # 👈 Set the parameter
def chat_model():
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    # GPU Enabled
    llm = LlamaCpp(
        model_path= MODEL_PATH,
        max_tokens=5000,#2000,
        n_ctx= 5000, #2048,
        temperature=0.75,
        f16_kv = True, # Must set to True, otherwise you will run into a problem after couple of calss
        top_p=1,
        callback_manager=callback_manager, 
        n_gpu_layers=100,
        verbose=False, # Verbose is required to pass to the callback manager
    )

    return llm

main_prompt = PromptTemplate(input_variables=['chat_history','user_input'], template=custom_prompt_template)
llm = chat_model()


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

if 'entity_memory' not in st.session_state:
    st.session_state.entity_memory = ConversationBufferMemory(input_key='user_input', memory_key='chat_history', return_messages=True)

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

LLM_Chain=LLMChain(prompt=main_prompt, memory=st.session_state.entity_memory,verbose=False, llm=llm)

# LLM_Chain = LLMChain(
#     llm=llm,
#     prompt=main_prompt,
#     verbose=False,
#     memory=memory,
#     llm_kwargs={"max_length": 4096}
# )

def response(prompt):
    res = LLM_Chain.run(prompt)
    return res

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking"):
            response = response(prompt)
            #response = LLM_Chain.run(prompt)
            #response = LLM_Chain.predict(user_input=prompt)
            placeholder = st.empty()
            full_response=''
            for item in response:
                full_response+=item
                sleep(0.01)
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role":"assistant", "content": full_response}
    st.session_state.messages.append(message)
