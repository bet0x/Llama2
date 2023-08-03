from langchain.document_loaders import PyPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from sentence_transformers import SentenceTransformer
from langchain.chains.question_answering import load_qa_chain

from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from huggingface_hub import hf_hub_download
from langchain.chains.question_answering import load_qa_chain

import pinecone
import os

def init_loader ():
    # Load the data
    path = r"C:/Users/Lukas/Desktop/My_Projects/To_Upload/Llama2/llama2_projects/llama2_pdf_chatbot_faiss_windows/data/Hotline_Wiki.pdf"
    loader = PyPDFLoader(path)
    data = loader.load()

    # Split the Texxt into chunks
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    docs=text_splitter.split_documents(data)

    return docs

def init_embeddings(index_name):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    query_result=embeddings.embed_query("Hello")
    len(query_result)

    # Create embedding for text chunk - can skip if you have already have an index
    #docsearch=Pinecone.from_texts([t.page_content for t in docs], embeddings, index_name=index_name)

    # Load an index
    docsearch = Pinecone.from_existing_index(index_name, embeddings)

    return docsearch

def init_pinecone():
    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', 'aa5d1b66-d1d9-451a-9f6b-dfa32db988fc')
    PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV', 'us-west1-gcp-free')

    pinecone.init( 
        api_key=PINECONE_API_KEY,  
        environment=PINECONE_API_ENV,  
    )
    index_name = "llama2-pdf-chatbox" 

    return index_name

def semantic_search(docsearch,query):
    docs=docsearch.similarity_search(query)
    return docs

def init_model(callback_manager):
    #model_path = r"C:/Users/jlukas/Desktop/llama-2-7b-chat.ggmlv3.q4_1.bin"
    model_path = r"D:/AI_CTS/Llama2/llama2_projects/llama2_quantized_models/7B_chat/llama-2-7b-chat.ggmlv3.q8_0.bin"
    n_gpu_layers = 40  # Change this value based on your model and your GPU VRAM pool.
    n_batch = 256  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

    llm = LlamaCpp(
        model_path=model_path,
        max_tokens=256,
        n_gpu_layers=40,
        n_batch= 512, #256,
        callback_manager=callback_manager,
        n_ctx= 1024,
        verbose=False,
        temperature=0.3,
    )

    return llm

def init_chain(llm):
    chain=load_qa_chain(llm,chain_type="stuff")
    return chain
    
def main():
    #docs       = init_loader()
    index      = init_pinecone()
    docssearch = init_embeddings(index)
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = init_model(callback_manager)

    chain = init_chain(llm)
    while True:
        query = input(f"Prompt: ")
        semantic_result  = semantic_search(docssearch,query)
        if query == "exit":
            print("exiting")
            break
        if query == "":
            continue
        
        llm_result = chain.run(input_documents =semantic_result, question=query)
        
        print(f"Answer: " +llm_result)

if __name__ == "__main__":
    main()