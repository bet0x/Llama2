from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

from langchain.vectorstores import Pinecone

# Use for CPU
from langchain.llms import CTransformers

# use for GPU
#from ctransformers import AutoModelForCausalLM

#from ctransformers.langchain import CTransformers

import os
import pinecone

import chainlit as cl
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

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

#Loading the model
def load_llm():
    # Load the locally downloaded model here
    
    llm = CTransformers(
        model = MODEL_PATH + "llama-2-7b-chat.ggmlv3.q8_0.bin",
        #model = MODEL_PATH + "orca-mini-3b.ggmlv3.q8_0.bin",
        #model = PATH + "airoboros-l2-7b-gpt4-1.4.1.ggmlv3.q8_0.bin",

        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
    )

    # Use this for GPU support
    #llm = AutoModelForCausalLM.from_pretrained(MODEL_PATH + "llama-2-7b-chat.ggmlv3.q8_0.bin", model_type='llama', gpu_layers=50)

    return llm

#QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cuda'})
    
    #db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    
    db = pinecone_init(embeddings)
    
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

#output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

#chainlit code
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to X-Fab Hotline Bot. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    #if sources:
    #    answer += f"\nSources:" + str(sources)
    #else:
    #    answer += "\nNo sources found"

    await cl.Message(content=answer).send()
