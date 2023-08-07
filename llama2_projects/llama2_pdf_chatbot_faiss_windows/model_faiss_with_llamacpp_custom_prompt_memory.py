from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory

# Use for CPU
#from langchain.llms import CTransformers
#from ctransformers.langchain import CTransformers

# Use for GPU
from langchain.llms import LlamaCpp

import chainlit as cl
import warnings

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

PATH = r"D:/AI_CTS/Llama2/llama2_projects/llama2_pdf_chatbot_faiss_windows/"

MODEL_PATH = r"D:/AI_CTS/Llama2/llama2_projects/llama2_quantized_models/7B_chat/llama-2-7b-chat.ggmlv3.q5_K_M.bin"

DB_FAISS_PATH = PATH + 'vectorstore/db_faiss'

# custom_prompt_template = """[INST] <<SYS>>
# You are helpful assistant, you always only answer for the assistant then you stop, read the chat history to get the context
# {context}

# {chat_history}

# Question: {question}
# Answer:
# <</SYS>>

# [/INST]"""


# custom_prompt_template = """[INST] <<SYS>>
# You are a helpful, respectful and expert engineer and scientist. Always answer the question as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

# If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer, just say that you don't know, ant submit your request to hotline@xfab.com

# {context}

# {chat_history}
# Question: {question}
# Answer:
# <</SYS>>

# [/INST]"""


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

CUSTOM_SYSTEM_PROMPT = """You are helpful assistant, you always only answer for the assistant then you stop, read the chat history to get the context"""
INSTRUCTION = "{context}\n\n Chat History:\n\n{chat_history} \n User:\n {question}"
SYSTEM_PROMPT = B_SYS + CUSTOM_SYSTEM_PROMPT + INSTRUCTION + E_SYS
custom_prompt_template = B_INST + SYSTEM_PROMPT + E_INST

print(custom_prompt_template)


def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(input_variables=['chat_history','context','question'],
                            template=custom_prompt_template)
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db, memory):
    chain_type_kwargs = {"prompt": prompt, "memory": memory}
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 1}),
                                       return_source_documents=True,
                                       chain_type_kwargs=chain_type_kwargs
                                       )
    return qa_chain

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

def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                        model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    memory = ConversationBufferMemory(input_key="question", memory_key="chat_history")
    qa = retrieval_qa_chain(llm, qa_prompt, db, memory)


    return qa

#output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result(query)
    return response

while True:
    query = input(f"\nPrompt: " )
    #semantic_result  = semantic_search(docssearch,query)
    if query == "exit":
        print("exiting")
        break
    if query == "":
        continue
    answer = final_result(query)
    
    
