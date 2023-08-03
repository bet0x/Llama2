from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Pinecone

from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from huggingface_hub import hf_hub_download


import os
import pinecone

import streamlit as st
from streamlit_chat import message

import warnings

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

PATH = r"C:/Users/Lukas/Desktop/My_Projects/To_Upload/Llama2/llama2_projects/llama2_pdf_chatbot_faiss_windows/"

MODEL_PATH = r"C:/Users/Lukas\Desktop/My_Projects/To_Upload/Llama2/llama2_projects/llama2_quantized_models/7B_chat/"
#MODEL_PATH = r"D:/AI_CTS/Llama2/llama2_projects/llama2_quantized_models/3B_Orca/"

DB_FAISS_PATH = PATH + 'vectorstore/db_faiss'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, ant submit your request to hotline@xfab.com

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

st.title("Hotline Virtual Assistant👩‍💼")
st.markdown("<h3 style='text-align: center; color: white;'>Built by <a href='https://github.com/AIAnytime'>X-Fab with ❤️ </a></h3>", unsafe_allow_html=True)

def pinecone_init(embeddings):
    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', 'aa5d1b66-d1d9-451a-9f6b-dfa32db988fc')
    PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV', 'us-west1-gcp-free')
    
    # initialize pinecone which can be copied from Pinecone 'Connect' button
    pinecone.init( 
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV,  # next to api key in console
    )
    index_name = "llama2-pdf-chatbox" # put in the name of your pinecone index here
    
    docsearch = Pinecone.from_existing_index(index_name, embeddings)
    
    return docsearch
    
def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

def load_llm():
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(
        model_path=MODEL_PATH + "llama-2-7b-chat.ggmlv3.q8_0.bin",
        max_tokens=256,
        n_gpu_layers=40,
        n_batch=256,
        callback_manager=callback_manager,
        n_ctx=1024,
        verbose=False,
    )

    return llm

def conversational_chat(query,embeddings,chain):
    docsearch = pinecone_init(embeddings)
 
    docs = docsearch.similarity_search(query)
    qa = chain.run(input_documents=docs, question=query)
    
    st.session_state(qa)
    
    #result = chain({"question": query, "chat_history": st.session_state['history']})
    #st.session_state['history'].append((query, result["answer"]))
    #return result["answer"]
    return qa

#QA Model Function
def qa_bot():

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cuda'})
    
    llm = load_llm()

    chain=load_qa_chain(llm, chain_type="stuff")
    # Faiss Database
    #db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    
    # Pinecone Database
  
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
    
    with container:
        with st.form(key='my_form', clear_on_submit=True):
            
            user_input = st.text_input("Query:", placeholder="Search your data here :", key='input')
            submit_button = st.form_submit_button(label='Send')
            
        if submit_button and user_input:
            output = conversational_chat(user_input,embeddings,chain)
            
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
    
qa_bot()
# while True:
#     query = input(f"Prompt: ")
#     x = qa_bot(query)

#     if query == "exit":
#         print("Exiting")
#     if query == "":
#         continue
#     print(x)
   
