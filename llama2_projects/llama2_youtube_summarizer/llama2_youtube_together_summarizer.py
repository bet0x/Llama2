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


MODEL_PATH = r"D:/AI_CTS/Llama2/llama2_projects/llama2_quantized_models/7B_chat/llama-2-7b-chat.ggmlv3.q4_K_M.bin"
PATH = r"C:/Users/Lukas/Desktop/My_Projects/To_Upload/Llama2/llama2_projects/llama2_youtube_summarizer/"

DATA_PATH = PATH + 'data/'
DB_FAISS_PATH = PATH + 'vectorstore/db_faiss'

def load_youtube_url():
    
    loader = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=128fp0rqfbE&ab_channel=TED-Ed", add_video_info=True)
    #loader = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=AGS45Fd9nmE&ab_channel=TheRisen", add_video_info=True)
    
    result = loader.load()
    print(f"Found Video From {result[0].metadata['author']} that is {result[0].metadata['length']} seconds length")
    print(result)

    return result

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

data = load_youtube_url()
result = split_chunking(data)

chain = load_summarize_chain(llm=llm, chain_type="map_reduce", verbose=True)
print(chain.run(result))