from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Pinecone

# Use for CPU
#from langchain.llms import CTransformers
#from ctransformers.langchain import CTransformers

# Use for GPU
from langchain.llms import LlamaCpp

import chainlit as cl
import warnings
import pinecone
import os

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

PATH = r"C:/Users/Lukas/Desktop/My_Projects/To_Upload/Llama2/llama2_projects/llama2_pdf_chatbot_faiss_windows/"

MODEL_PATH = r"D:/llama2_quantized_models/7B_chat/llama-2-7b-chat.ggmlv3.q5_K_M.bin"

DB_FAISS_PATH = PATH + 'vectorstore/db_faiss'

# custom_prompt_template = """[INST] <<SYS>>
# You are a helpful, respectful and expert engineer and scientist. Always answer the question as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

# If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer, just say that you don't know, ant submit your request to hotline@xfab.com

# {history}
# {context}

# {question}
# <</SYS>>

# [/INST]"""

custom_prompt_template = """[INST] <<SYS>>
You are helpful assistant, you always only answer for the assistant then you stop, read the chat history to get the context

{context}

{chat_history}
Question: {user_input}
<</SYS>>

[/INST]"""

print(custom_prompt_template)

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(input_variables=['chat_history','context', 'user_input'],
                            template=custom_prompt_template)
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db, memory):
    chain_type_kwargs = {"prompt": prompt, "memory": memory}
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs=chain_type_kwargs
                                       )
    return qa_chain

#Loading the model
def load_llm():
    
    # Use CUDA GPU
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(
        model_path= MODEL_PATH,
        max_tokens=256,
        n_gpu_layers=35,
        n_batch= 512, #256,
        callback_manager=callback_manager,
        n_ctx= 1024,
        verbose=False,
        temperature=0.2,
    )

    return llm

def init_pinecone():
    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', 'aa5d1b66-d1d9-451a-9f6b-dfa32db988fc')
    PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV', 'us-west1-gcp-free')

    pinecone.init( 
        api_key=PINECONE_API_KEY,  
        environment=PINECONE_API_ENV,  
    )
    index_name = "llama2-pdf-chatbox" 
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    docsearch = Pinecone.from_existing_index(index_name, embeddings)

    return docsearch

def semantic_search(docsearch,query):
    docs=docsearch.similarity_search(query)
    return docs

#QA Model Function
def qa_bot(ask):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                        model_kwargs={'device': 'cpu'})
    #db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    
    db = init_pinecone()
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    memory = ConversationBufferMemory(input_key="user_input", memory_key="chat_history")
    
    # Semantic Search
    semantic  = semantic_search(db,ask)
    
    # AI Search
    qa = retrieval_qa_chain(llm, qa_prompt, db, memory)
    print(semantic)

    return qa

#output function
def final_result(query):
    qa_result = qa_bot(query)
    response = qa_result(query)
    return response

while True:
    query = input(f"\nPrompt: " )
    if query == "exit":
        print("exiting")
        break
    if query == "":
        continue
    answer = final_result(query)
    
    
