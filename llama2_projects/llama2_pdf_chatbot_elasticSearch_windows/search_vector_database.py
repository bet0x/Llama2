from langchain import ElasticVectorSearch
from langchain.document_loaders import PyPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.chains.question_answering import load_qa_chain
from elasticsearch import Elasticsearch
from langchain.chains import RetrievalQA

PATH = r"D:/AI_CTS/Llama2/llama2_projects/llama2_pdf_chatbot_faiss_windows/"

DATA_PATH = PATH + 'data/'
DB_FAISS_PATH = PATH + 'vectorstore/db_faiss'

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

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
    
#print(db.client.info())

query = "WHat is FLATPV ?"
docs = db.similarity_search(query)

#print(docs)
print(docs[0].page_content)

