from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import WebBaseLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import nest_asyncio
nest_asyncio.apply()

PATH = r"D:/AI_CTS/Llama2/llama2_projects/llama2_chat_with_webloader/"
DATA_PATH = PATH + 'data/'
DB_FAISS_PATH = PATH + 'vectorstore/db_faiss'

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Create vector database
def create_vector_db():
    web = [
        "http://www.tanyajpeterson.com/8-mindful-lessons-in-wellbeing-i-learned-from-my-frog/",
        "https://www.thechurchnews.com/leaders/2023/8/28/23847533/elder-cook-byu-university-conference-church-doctrinal-purposes-education"]
    
    loader = WebBaseLoader(web)
    
    # For multiple Url
    loader.requests_per_second = 1
    documents = loader.aload()
    
    # For single Url
    #documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    create_vector_db()