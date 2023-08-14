from langchain.document_loaders import PyPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate

from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from huggingface_hub import hf_hub_download
from langchain.chains.question_answering import load_qa_chain

import pinecone
import os

# template = """[INST] <<SYS>>
# You are chat customer support agent, you're helpful and respectful. Always answer the question as helpfully as possible. Use the question context to answer the question at the end.

# If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question please ask customer to submit their request to hotline@xfab.com

# {context}

# {chat_history}
# Question: {user_input}
# Answer:
# <</SYS>>

# [/INST]"""

# template = """[INST] <<SYS>>
# Your name is Dmitry, You are helpful assistant, you always only answer for the assistant then you stop, read the chat history to get the context.
# If you don't know the answer, just say you don't know and submit the request to hotline@xfab.com for further assistance.
# <</SYS>>
# {context}

# {chat_history}
# Question: {user_input}
# [/INST]"""

template = """[INST] <<SYS>>
Your name is Kelly, You are foundry technologies expert and very helpful assistant, you always open and only answer for the question then you stop, read the chat history to get the context.
If you don't know the answer, just say you don't know and submit the request to hotline@xfab.com for further assistance.
<</SYS>>
{context}

{chat_history}
Question: {user_input}
[/INST]"""

print(template)

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
    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', '700dbf29-7b1d-435b-9da1-c242f7a206e6')
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

def init_model():
    
    model_path = r"D:/llama2_quantized_models/7B_chat/llama-2-7b-chat.ggmlv3.q5_K_M.bin"

    n_gpu_layers = 32  # Change this value based on your model and your GPU VRAM pool.
    n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    llm = LlamaCpp(
        model_path=model_path,
        max_tokens=256,
        n_gpu_layers=35,
        n_batch= 512, #256,
        callback_manager=callback_manager,
        n_ctx= 2048, #1024, - Increase this to add context length
        verbose=True,
        temperature=0,
    )

    return llm

def init_chain(llm,prompt,memory):
    chain=load_qa_chain(llm, chain_type="stuff",memory=memory, prompt=prompt)
    return chain

def main():
    #docs  = init_loader()
    index  = init_pinecone()

    prompt = PromptTemplate(input_variables=["chat_history","user_input","context"], template=template)
    memory = ConversationBufferMemory(input_key="user_input",memory_key="chat_history",)

    docssearch = init_embeddings(index)
    llm = init_model()

    chain = init_chain(llm,prompt,memory)

    while True:
        query = input(f"\nPrompt: " )
        semantic_result  = semantic_search(docssearch,query)
        if query == "exit":
            print("exiting")
            break
        if query == "":
            continue

        chain_input={
            "input_documents":semantic_result,
            "user_input":query
        }
        llm_result = chain(chain_input,return_only_outputs=True)
        llm_result['output_text']

        #llm_result = chain.run(input_documents =semantic_result, question=query)
        #print(f"Answer: " +llm_result)

if __name__ == "__main__":
    main()