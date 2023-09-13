import gradio as gr
from langchain import PromptTemplate
from langchain import LLMChain
from langchain.memory import ConversationBufferMemory

from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE

from langchain.llms import LlamaCpp
from time import sleep
import streamlit as st

from togetherllm import TogetherLLM

st.set_page_config(page_title='🧠MemoryBot🤖', layout='wide')
st.title('🎈 Llama Code 2 Chatbot | Entity Memory')

llm = TogetherLLM(
    model= "togethercomputer/CodeLlama-7b-Instruct",
    temperature=0.7,
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
    st.title("Code Wiki")
    st.header("Settings")
    add_replicate_api = st.text_input("Enter your password here", type='password')

    if not (add_replicate_api.startswith('jl')):
        st.warning('Please enter your credentials', icon='⚠️')
    else:
        st.success('Proceed to entering your prompt message!', icon='👉')

if 'entity_memory' not in st.session_state:
    st.session_state.entity_memory = ConversationEntityMemory(llm=llm, k=3 )
   
llm_chain=LLMChain(prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE, memory=st.session_state.entity_memory, llm=llm ,verbose=True)

if prompt := st.chat_input(st.session_state["input"], disabled=not add_replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

def generate_response(prompt):
    if prompt:
        output = llm_chain.run(prompt)
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
            #response = generate_response(prompt)
            response = llm_chain.run(prompt)
            st.session_state.past.append(prompt)
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




