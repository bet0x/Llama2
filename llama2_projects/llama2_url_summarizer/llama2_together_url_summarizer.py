"""
Library:
pip install youtube-transcript-api
pip install pytube
"""
from langchain import PromptTemplate
from langchain import LLMChain
from langchain.document_loaders import YoutubeLoader
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import textwrap

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from togetherllm import TogetherLLM

################################################################
from langchain.document_loaders import UnstructuredURLLoader
from langchain.document_loaders import SeleniumURLLoader

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.vectorstores import Pinecone
import pinecone
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from langchain import HuggingFacePipeline
from huggingface_hub import notebook_login
import textwrap
import sys
import os
import torch
from langchain.document_loaders import WebBaseLoader, DirectoryLoader


MODEL_PATH = r"D:/AI_CTS/Llama2/llama2_projects/llama2_quantized_models/7B_chat/llama-2-7b-chat.ggmlv3.q4_K_M.bin"
PATH = r"C:/Users/Lukas/Desktop/My_Projects/To_Upload/Llama2/llama2_projects/llama2_youtube_summarizer/"

DATA_PATH = PATH + 'data/'
DB_FAISS_PATH = PATH + 'vectorstore/db_faiss'

def load_website_url():
    URLs=[
        'https://www.thechurchnews.com/leaders/2023/8/28/23847533/elder-cook-byu-university-conference-church-doctrinal-purposes-education',
        'http://www.tanyajpeterson.com/8-mindful-lessons-in-wellbeing-i-learned-from-my-frog/',
        'https://www.churchofjesuschrist.org/learn/youth-theme-2023?lang=eng'
    ]
        
    #loaders=UnstructuredURLLoader(urls=URLs)
    loaders=SeleniumURLLoader(urls=URLs)
    data=loaders.load()

    return data
    
# We need to split the youtube result because due to limited input token - ValueError: Requested tokens (6046) exceed context window of 2048
def split_chunking(result):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50)
    texts = text_splitter.split_documents(result)

    return texts

llm = TogetherLLM(
    model= "togethercomputer/llama-2-7b-chat",
    temperature=0.7,
    max_tokens=512
) 

data = load_website_url()
result = split_chunking(data)

chain = load_summarize_chain(llm=llm, chain_type="map_reduce", verbose=True)
print(chain.run(result))