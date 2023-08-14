from langchain.vectorstores import ElasticVectorSearch
from langchain.document_loaders import PyPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.chains.question_answering import load_qa_chain
from elasticsearch import Elasticsearch
from langchain.vectorstores import FAISS


# Convert to Embedded
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def create_vector_db():
    #path = r"D:/AI_CTS/Llama2/llama2_projects/llama2_pdf_chatbot_faiss_windows/data//V1/Hotline_Wiki.pdf"
    path = r"D:/AI_CTS/Llama2/llama2_projects/llama2_pdf_chatbot_faiss_windows/data/V2/Hotline_Wiki_v2.pdf"

    # Load the Data
    loader = PyPDFLoader(path)
    data = loader.load()

    # Split the text into Chunks
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    docs=text_splitter.split_documents(data)

    CERT_FINGERPRINT = "7e73d3cf8918662a27be6ac5f493bf55bd8af2a95338b9b8c49384650c59db08"
    CERT_PATH = "D:\elasticsearch-8.4.2\config\certs\http_ca.crt"

    ELASTIC_PASSWORD = "Eldernangkai92"

    elasticsearch_url = f"https://elastic:Eldernangkai92@localhost:9200"
    db= ElasticVectorSearch.from_documents(
        docs,
        embeddings,
        elasticsearch_url=elasticsearch_url,
        index_name="elastic_wiki",
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
    create_vector_db()

