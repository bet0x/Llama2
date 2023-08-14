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

MODEL_PATH = r"D:/llama2_quantized_models/7B_chat/llama-2-7b-chat.ggmlv3.q5_K_M.bin"

# custom_prompt_template = """[INST] <<SYS>>
# You are a helpful, respectful and expert engineer and scientist. Always answer the question as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

# If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer, just say that you don't know, ant submit your request to hotline@xfab.com

# {chat_history}
# {context}

# {question}
# <</SYS>>
# [/INST]"""

# custom_prompt_template = """[INST] <<SYS>>
# Your name is Dmitry, You are helpful assistant, you always only answer for the assistant then you stop, read the chat history to get the context.
# If you don't know the answer, just say you don't know and submit the request to hotline@xfab.com for further assistance.
# <</SYS>>
# {context}

# {chat_history}
# Question: {question}
# [/INST]"""

custom_prompt_template = """[INST] <<SYS>>
Your name is Kelly, You are foundry technologies expert and very helpful assistant, you always open and only answer for the question professionally.
If you don't know the answer, just say you don't know and submit the request to hotline@xfab.com for further assistance.
<</SYS>>
{context}

{chat_history}
Question: {question}
[/INST]"""

print(custom_prompt_template)

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(input_variables=['chat_history','context', 'question'],
                            template=custom_prompt_template)
    return prompt


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
    temperature=0,
)
    
#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db, memory):
    chain_type_kwargs = {"prompt": prompt, "memory": memory}
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(), #search_kwargs={'k': 1}
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
        n_ctx= 2048, #1024, - Increase this to add context length1024,
        verbose=True,
        temperature=0,
    )

    return llm

def init_pinecone():
    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', '700dbf29-7b1d-435b-9da1-c242f7a206e6')
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
    db = init_pinecone()
    #llm = load_llm()
    
    qa_prompt = set_custom_prompt()
    memory = ConversationBufferMemory(input_key="question", memory_key="chat_history")
    
    # Semantic Search
    semantic  = semantic_search(db,ask)
    
    # AI Search
    qa = retrieval_qa_chain(llm, qa_prompt, db, memory)
    print(f"""{semantic} \n\n""")

    return qa

def final_result(query):
    qa_result = qa_bot(query)
    response = qa_result(query)
    return response

while True:
    query = input(f"\n\nPrompt: " )
    if query == "exit":
        print("exiting")
        break
    if query == "":
        continue
    answer = final_result(query)
    
    
