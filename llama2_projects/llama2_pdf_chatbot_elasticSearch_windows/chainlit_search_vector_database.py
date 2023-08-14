import chainlit as cl
from langchain import ElasticVectorSearch
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

CERT_PATH = "D:\elasticsearch-8.4.2\config\certs\http_ca.crt"
ELASTIC_PASSWORD = "Eldernangkai92"

elasticsearch_url = f"https://elastic:Eldernangkai92@localhost:9200"

CERT_FINGERPRINT = "7e73d3cf8918662a27be6ac5f493bf55bd8af2a95338b9b8c49384650c59db08"

# Initial message to start - This function will start first before others
@cl.on_chat_start
async def start():
    # 1st - This message will get display first
    msg = cl.Message(content="Starting the bot...")
    await msg.send()

    # 2nd - Then we update the content of the message
    msg.content = "Hi, Welcome to X-Fab Hotline Bot. What is your query ?"
    await msg.send()

@cl.on_message
async def main(message: str):
    result = message
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
    docs = db.similarity_search(result, k=3)
    print(docs)
    
    for i in range(len(docs)):
        await cl.Message(content=f"Sure, here is the message {docs[i].page_content}").send()
    
    #await cl.Message(content=f"Sure, here is the message {docs[0].page_content}").send()
