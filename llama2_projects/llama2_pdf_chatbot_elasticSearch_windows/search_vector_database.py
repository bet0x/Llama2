from langchain import ElasticVectorSearch
from langchain.document_loaders import PyPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.chains.question_answering import load_qa_chain
from elasticsearch import Elasticsearch
from langchain.chains import RetrievalQA


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

CERT_PATH = "D:\elasticsearch-8.4.2\config\certs\http_ca.crt"
ELASTIC_PASSWORD = "Eldernangkai92"

elasticsearch_url = f"https://elastic:Eldernangkai92@localhost:9200"

CERT_FINGERPRINT = "7e73d3cf8918662a27be6ac5f493bf55bd8af2a95338b9b8c49384650c59db08"

db= ElasticVectorSearch(
        elasticsearch_url=elasticsearch_url,
        index_name="elastic_wiki",
        ssl_verify={
            "verify_certs": True,
            "basic_auth": ("elastic", ELASTIC_PASSWORD), 
            "ssl_assert_fingerprint" :  CERT_FINGERPRINT # You can use fingerprint also
            #"ca_certs": CERT_PATH, # You can use certificate path as well
        },
        embedding=embeddings
        )
    
#print(db.client.info())

query = "how to setup pdk using json ?"
docs = db.similarity_search(query)

#print(docs)
print(docs[0].page_content)

