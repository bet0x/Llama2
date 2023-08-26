import streamlit as st
from st_pages import Page, Section, show_pages, add_page_title
from st_pages import show_pages_from_config

from langchain import PromptTemplate, LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from streamlit_chat import message
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

st.title('🎈 Llama 2 Chatbot | Conversation Chain with Memory')

llm = TogetherLLM(
model= "togethercomputer/llama-2-7b-chat",
temperature=0,
max_tokens=512
)

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

# Define function to start a new chat
def new_chat():
    """
    Clears session state and starts a new chat.
    """
    save = []
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        save.append("User:" + st.session_state["past"][i])
        save.append("Bot:" + st.session_state["generated"][i])        
    st.session_state["stored_session"].append(save)
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["input"] = ""
    st.session_state.entity_memory.entity_store = {}
    st.session_state.entity_memory.buffer.clear()

# Set up sidebar with various options
with st.sidebar.expander("🛠️ ", expanded=False):
    st.title("X-Chat")
    st.header("Settings")
    add_replicate_api = st.text_input("Enter your password here", type='password')

    if not (add_replicate_api.startswith('jl')):
        st.warning('Please enter your credentials', icon='⚠️')
    else:
        st.success('Proceed to entering your prompt message!', icon='👉')
 
    # Option to preview memory store
    if st.checkbox("Preview memory store"):
        #with st.expander("Memory-Store", expanded=False):
        st.info(st.session_state.entity_memory.store)
        
    # Option to preview memory buffer
    if st.checkbox("Preview memory buffer"):
        #with st.expander("Bufffer-Store", expanded=False):
        st.info(st.session_state.entity_memory.buffer)
        
    K = st.number_input(' (#)Summary of prompts to consider',min_value=3,max_value=1000)
    
# Create a ConversationEntityMemory object if not already created
if 'entity_memory' not in st.session_state:
    st.session_state.entity_memory = ConversationEntityMemory(llm=llm, k=K )
    
Conversation = ConversationChain(
    llm=llm,
    prompt = ENTITY_MEMORY_CONVERSATION_TEMPLATE,
    memory = st.session_state.entity_memory
)

# Add a button to start a new chat
st.sidebar.button("New Chat", on_click = new_chat, type='primary')

user_input = get_text()

if user_input:
    output = Conversation.run(input=user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)
    
# Allow to download as well
download_str = []
# Display the conversation history using an expander, and allow the user to download it
with st.expander("Conversation", expanded=True):
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        st.info(st.session_state["past"][i],icon="🧐")
        st.success(st.session_state["generated"][i], icon="🤖")
        download_str.append(st.session_state["past"][i])
        download_str.append(st.session_state["generated"][i])
    
    # Can throw error - requires fix
    download_str = '\n'.join(download_str)
    if download_str:
        st.download_button('Download',download_str)
        
# Display stored conversation sessions in the sidebar
for i, sublist in enumerate(st.session_state.stored_session):
        with st.sidebar.expander(label= f"Conversation-Session:{i}"):
            st.write(sublist)

# Allow the user to clear all stored conversation sessions
if st.session_state.stored_session:   
    if st.sidebar.checkbox("Clear-all"):
        del st.session_state.stored_session