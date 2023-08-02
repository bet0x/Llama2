import streamlit as st
from streamlit_chat import message
from langchain import ElasticVectorSearch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.question_answering import load_qa_chain

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

def load_model():
    model_path = r"D:/AI_CTS/Llama2/llama2_projects/llama2_quantized_models/7B_chat/llama-2-7b-chat.ggmlv3.q8_0.bin"
    #model_path = r"C:/Users/Lukas\Desktop/My_Projects/To_Upload/Llama2/llama2_projects/llama2_quantized_models/7B_chat/llama-2-7b-chat.ggmlv3.q8_0.bin"

    n_gpu_layers = 40  # Change this value based on your model and your GPU VRAM pool.
    n_batch = 256  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

    # Loading model,
    llm = LlamaCpp(
        model_path=model_path,
        max_tokens=256,
        n_gpu_layers=40,
        n_batch= 512, #256,
        callback_manager=callback_manager,
        n_ctx= 1024,
        verbose=False,
    )
    return llm

def load_db(embeddings):
    CERT_PATH = "D:\elasticsearch-8.4.2\config\certs\http_ca.crt"
    ELASTIC_PASSWORD = "Eldernangkai92"
    elasticsearch_url = f"https://elastic:Eldernangkai92@localhost:9200"
    db= ElasticVectorSearch(
        elasticsearch_url=elasticsearch_url,
        index_name="elastic_vector",
        ssl_verify={
            "verify_certs": True,
            "basic_auth": ("elastic", ELASTIC_PASSWORD),
            "ca_certs": CERT_PATH,
        },
        embedding=embeddings
    )
    return db

def semantic_search(query):
    # Using Llama 2
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                            model_kwargs={'device': 'cpu'})
    #llm = load_llm()

    #Using Semantic Search
    db = load_db(embeddings)
    docs = db.similarity_search(query)
    x = (docs[0].page_content)
    print(x)

def llama2_search(query):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                            model_kwargs={'device': 'cpu'})
    
    db = load_db(embeddings)
    llm = load_model()
    chain = load_qa_chain(llm, chain_type="stuff")

    docs=db.similarity_search(query)
    result = chain.run(input_documents=docs, question=query)
    print(result)

semantic_search("What is Flatpv ?")
llama2_search("What is Flatpv ? ")
