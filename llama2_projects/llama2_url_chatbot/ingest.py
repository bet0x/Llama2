from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import WebBaseLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import nest_asyncio
nest_asyncio.apply()

PATH = r"D:/AI_CTS/Llama2/llama2_projects/llama2_url_chatbot/"

DATA_PATH = PATH + 'data/'
DB_FAISS_PATH = PATH + 'vectorstore/db_faiss'

# Create vector database
def create_vector_db():
    web = [
        "https://raw.githubusercontent.com/basecamp/handbook/master/titles-for-programmers.md",
        "https://raw.githubusercontent.com/develtechmon/Llama2/main/Processing_Tools/Data_Set_Json_To_MarkDown_Converter/output.md"]
    
    loader = WebBaseLoader(web)
    
    # For multiple Url
    loader.requests_per_second = 1
    documents = loader.aload()
    
    # For single Url
    #documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})
    
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    create_vector_db()