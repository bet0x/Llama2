from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import ElasticVectorSearch

# Use for CPU
#from langchain.llms import CTransformers
#from ctransformers.langchain import CTransformers

# Use for GPU
from langchain.llms import LlamaCpp

import chainlit as cl
import warnings
import pinecone
import os

import sys
sys.setrecursionlimit(1500)

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

#MODEL_PATH = r"D:/llama2_quantized_models/7B_chat/llama-2-7b-chat.ggmlv3.q5_K_M.bin"
MODEL_PATH = r"D:/AI_CTS/Llama2/llama2_projects/llama2_quantized_models/7B_chat/llama-2-7b-chat.ggmlv3.q5_K_M.bin"

custom_prompt_template = """[INST] <<SYS>>
Your name is Kelly, You are helpful assistant, you always open and only answer for the assistant then you stop, read the chat history to get the context.
If you don't know the answer, just say you don't know and submit the request to hotline@xfab.com for further assistance.
<</SYS>>
{context}

{chat_history}
Question: {question}
[/INST]"""

print(custom_prompt_template)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                            model_kwargs={'device': 'cpu'})

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

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(input_variables=['chat_history','context', 'question'],
                            template=custom_prompt_template)
    return prompt


def init_llm():
    # Use CUDA GPU
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(
        model_path= MODEL_PATH,
        max_tokens=256,
        n_gpu_layers=35,
        n_batch= 512, #256,
        callback_manager=callback_manager,
        n_ctx= 2048,
        f16_kv=True,
        verbose=False,
        temperature=0.2,
    )
    return llm
    
def conversationalretrieval_qa_chain(llm, prompt, db, memory):
    chain_type_kwargs = {"prompt": prompt}
    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                     chain_type= 'stuff',
                                                     #retriever=db.as_retriever(search_kwargs={'k': 3}),
                                                     retriever=db.as_retriever(),
                                                     verbose=False,
                                                     memory=memory,
                                                     combine_docs_chain_kwargs=chain_type_kwargs
                                                     )
    return qa_chain

    
def load_db():
    CERT_FINGERPRINT = "7e73d3cf8918662a27be6ac5f493bf55bd8af2a95338b9b8c49384650c59db08"
    CERT_PATH = "D:\elasticsearch-8.4.2\config\certs\http_ca.crt"
    ELASTIC_PASSWORD = "Eldernangkai92"
    elasticsearch_url = f"https://elastic:Eldernangkai92@localhost:9200"

    db= ElasticVectorSearch(
        elasticsearch_url=elasticsearch_url,
        index_name="elastic_wiki",
        ssl_verify={
            "verify_certs": True,
            "basic_auth": ("elastic", ELASTIC_PASSWORD),
            "ssl_assert_fingerprint" :  CERT_FINGERPRINT # You can use fingerprint also
            #"ca_certs": CERT_PATH, # You can Certificate path too
        },
        embedding=embeddings
    )
    
    return db


def semantic_search(docsearch,query):
    docs=docsearch.similarity_search(query)
    return docs

#QA Model Function
def qa_bot(ask):

    llm = init_llm()
    db = load_db()
    #db = semantic_search(docs, ask)
    
    qa_prompt = set_custom_prompt()
    memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    qa = conversationalretrieval_qa_chain(llm, qa_prompt, db, memory)
    
    result = qa({"question": ask})
    res = result['answer']

    return res

#output function
def final_result(query):
    response = qa_bot(query)
    return response

while True:
    query = input(f"\n\nPrompt: " )
    if query == "exit":
        print("exiting")
        break
    if query == "":
        continue
    answer = final_result(query)
    
    
