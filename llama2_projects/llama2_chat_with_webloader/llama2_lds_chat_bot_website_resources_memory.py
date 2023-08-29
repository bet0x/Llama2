
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter

from langchain.vectorstores import FAISS
from langchain.vectorstores import Pinecone
import pinecone

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings import HuggingFaceEmbeddings
import textwrap
import sys
import os
import torch
from togetherllm import TogetherLLM
from time import sleep
import streamlit as st
from langchain import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from streamlit_chat import message

PATH = r"D:/AI_CTS/Llama2/llama2_projects/llama2_chat_with_webloader/"
DATA_PATH = PATH + 'data/'
DB_FAISS_PATH = PATH + 'vectorstore/db_faiss'

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
llm = TogetherLLM(
model= "togethercomputer/llama-2-7b-chat",
temperature=0,
max_tokens=512
)

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
if 'history' not in st.session_state:
    st.session_state['history'] = []

with st.sidebar:
    st.title("LDS Chat")
    st.header("Settings")
    add_replicate_api = st.text_input("Enter your password here", type='password')

    if not (add_replicate_api.startswith('jl')):
        st.warning('Please enter your credentials', icon='⚠️')
    else:
        st.success('Proceed to entering your prompt message!', icon='👉')

if prompt := st.chat_input(st.session_state["input"], disabled=not add_replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]
    
# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(input_variables=['chat_history','context', 'question'],
                            template=custom_prompt_template)
    return prompt

if 'entity_memory' not in st.session_state:
    st.session_state.entity_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def conversationalretrieval_qa_chain(llm,qa_prompt,db):
    chain_type_kwargs = {"prompt": qa_prompt}
    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm,chain_type='stuff',
                                                retriever=db.as_retriever(search_kwargs={"k":2}),
                                                verbose=True,
                                                memory=st.session_state.entity_memory,
                                                combine_docs_chain_kwargs=chain_type_kwargs)

    return qa_chain

def conversation_chat(query):
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    qa_prompt = set_custom_prompt()
    qa = conversationalretrieval_qa_chain(llm,qa_prompt,db)

    #result = qa({"question": query, "chat_history": st.session_state['history']})
    result = qa({"question": query})

    #st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

#Clear the chat messages
def clear_chat_history():
   st.session_state.messages=[{"role": "assistant","content": "How may i assist you today ?"}]

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking"):
            response = conversation_chat(prompt)
            
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







# st.set_page_config(page_title='🧠MemoryBot🤖', layout='wide')

# st.title('🎈 Llama 2 Chatbot | Entity Memory')



# with st.sidebar:
#     st.title("X-Chat")
#     st.header("Settings")
#     add_replicate_api = st.text_input("Enter your password here", type='password')

#     if not (add_replicate_api.startswith('jl')):
#         st.warning('Please enter your credentials', icon='⚠️')
#     else:
#         st.success('Proceed to entering your prompt message!', icon='👉')

# if ask := st.chat_input(disabled=not add_replicate_api):
#     with st.chat_message("user"):
#         st.write(ask)




# def initialize_session_state():
#     if 'history' not in st.session_state:
#         st.session_state['history'] = []

#     if 'generated' not in st.session_state:
#         st.session_state['generated'] = ["Hello! Ask me anything about 🤗"]

#     if 'past' not in st.session_state:
#         st.session_state['past'] = ["Hey! 👋"]
        

# def display_chat_history():
#     reply_container = st.container()
#     container = st.container()

#     with container:
#         with st.form(key='my_form', clear_on_submit=True):
#             user_input = st.text_input("Question:", placeholder="Ask about your Mental Health", key='input')
#             submit_button = st.form_submit_button(label='Send')

#         if submit_button and user_input:
#             output = conversation_chat(user_input)

#             st.session_state['past'].append(user_input)
#             st.session_state['generated'].append(output)

#     if st.session_state['generated']:
#         with reply_container:
#             for i in range(len(st.session_state['generated'])):
#                 message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
#                 message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

# # Initialize session state
# initialize_session_state()
# # Display chat history
# display_chat_history()