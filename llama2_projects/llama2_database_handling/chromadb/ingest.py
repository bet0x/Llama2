import os
from langchain.vectorstores import Pinecone
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
import pinecone
import os
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
path = r"D:/AI_CTS/Llama2/Processing_Tools/Data_Set_Json_To_Txt/data_split/"

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', '700dbf29-7b1d-435b-9da1-c242f7a206e6')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV', 'us-west1-gcp-free')

def create_vector_db():

    # Multiple Text FIle
    loader = DirectoryLoader(path, glob='*.txt', loader_cls=TextLoader)
    
    # Single Text File
    #loader = TextLoader(path + "output_0_00001.txt")
    data = loader.load()

    print(data)

    query_result=embeddings.embed_query("Hello")
    dimensions = len(query_result)
    print(dimensions) # 768 for mpne5-base-v2 - therefore we need to create a database with 768

    # initialize pinecone which can be copied from Pinecone 'Connect' button
    pinecone.init( 
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV,  # next to api key in console
    )
    index_name = "new-wikidb-v1" # put in the name of your pinecone index here 

    docsearch=Pinecone.from_texts([t.page_content for t in data], embeddings, index_name=index_name)

    query = "Where i can fatal violations list ?"
    docs=docsearch.similarity_search(query)
    print(docs)

if __name__ == "__main__":
    create_vector_db()