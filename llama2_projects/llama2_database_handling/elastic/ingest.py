from langchain.vectorstores import ElasticVectorSearch
from langchain.document_loaders import PyPDFLoader, OnlinePDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.chains.question_answering import load_qa_chain
from elasticsearch import Elasticsearch
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader


# Convert to Embedded
#embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/msmarco-MiniLM-L-12-v3")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
path = r"D:/AI_CTS/Llama2/Processing_Tools/Data_Set_Json_To_Txt/data_split/"

def create_text_vector_db():
    # Multiple Text FIle
    loader = DirectoryLoader(path, glob='*.txt', loader_cls=TextLoader)
    
    # Single Text File
    #loader = TextLoader(path + "output_0_00001.txt")
    data = loader.load()

    print(data)

    query_result=embeddings.embed_query("Hello")
    dimensions = len(query_result)
    print(dimensions) # 768 for mpne5-base-v2 - therefore we need to create a database with 768

    # Split the text into Chunks
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs=text_splitter.split_documents(data)

    CERT_FINGERPRINT = "7e73d3cf8918662a27be6ac5f493bf55bd8af2a95338b9b8c49384650c59db08"
    CERT_PATH = "D:\elasticsearch-8.4.2\config\certs\http_ca.crt"

    ELASTIC_PASSWORD = "Eldernangkai92"

    elasticsearch_url = f"https://elastic:Eldernangkai92@localhost:9200"
    db= ElasticVectorSearch.from_documents(
        docs,
        embeddings,
        elasticsearch_url=elasticsearch_url,
        index_name="new_wikidb",
        ssl_verify={
            "verify_certs": True,
            "basic_auth": ("elastic", ELASTIC_PASSWORD), # You can use fingerprint also
            "ssl_assert_fingerprint" : CERT_FINGERPRINT, # You can use certificate path as well
            #"ca_certs": CERT_PATH,
        }
        )
    
    print(db.client.info())
    
    query = "What is FLATPV ?"
    docs = db.similarity_search(query)
    print(docs)



def create_vector_db():
    #path = r"D:/AI_CTS/Llama2/llama2_projects/llama2_pdf_chatbot_faiss_windows/data//V1/Hotline_Wiki.pdf"
    path = r"D:/AI_CTS/Llama2/llama2_projects/llama2_pdf_chatbot_faiss_windows/data/V3/Hotline_Wiki_v3.pdf"

    # Load the Data
    loader = PyPDFLoader(path)
    data = loader.load()

    query_result=embeddings.embed_query("Hello")
    dimensions = len(query_result)
    print(dimensions) # 768 for mpne5-base-v2 - therefore we need to create a database with 768

    # Split the text into Chunks
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs=text_splitter.split_documents(data)

    CERT_FINGERPRINT = "7e73d3cf8918662a27be6ac5f493bf55bd8af2a95338b9b8c49384650c59db08"
    CERT_PATH = "D:\elasticsearch-8.4.2\config\certs\http_ca.crt"

    ELASTIC_PASSWORD = "Eldernangkai92"

    elasticsearch_url = f"https://elastic:Eldernangkai92@localhost:9200"
    db= ElasticVectorSearch.from_documents(
        docs,
        embeddings,
        elasticsearch_url=elasticsearch_url,
        index_name="new_wikidb",
        ssl_verify={
            "verify_certs": True,
            "basic_auth": ("elastic", ELASTIC_PASSWORD), # You can use fingerprint also
            "ssl_assert_fingerprint" : CERT_FINGERPRINT, # You can use certificate path as well
            #"ca_certs": CERT_PATH,
        }
        )
    
    print(db.client.info())
    
    query = "What is FLATPV ?"
    docs = db.similarity_search(query)
    print(docs)
    
if __name__ == "__main__":
    #create_vector_db()
    create_text_vector_db

