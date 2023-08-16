from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain import PromptTemplate, LLMChain
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings

from togetherllm import TogetherLLM
import together
from langchain.vectorstores import Pinecone
import pinecone
import os
from langchain.embeddings import HuggingFaceEmbeddings
import chainlit as cl

#embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
#                                                      model_kwargs={"device": "cuda"})

PATH = r"C:/Users/Lukas/Desktop/My_Projects/To_Upload/Llama2/llama2_projects/llama2_together_api/data/"
DB_PATH = r"C:/Users/Lukas/Desktop/My_Projects/To_Upload/Llama2/llama2_projects/llama2_together_api/"

#read the chat history to get the context.
custom_prompt_template = """[INST] <<SYS>>
Your name is Kelly, You are semiconductor foundry expert and a very helpful assistant, you always open to the question and only answer for the question.
If you don't know the answer, just say you don't know and submit the request to hotline@xfab.com for further assistance.
<</SYS>>
{context}

{chat_history}
Question: {question}
[/INST]"""

print(custom_prompt_template)

llm = TogetherLLM(
    model= "togethercomputer/llama-2-7b-chat",
    temperature=0,
    max_tokens=512
)

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(input_variables=['chat_history','context', 'question'],
                            template=custom_prompt_template)
    return prompt

def conversationalretrieval_qa_chain(llm, prompt, db, memory):
    chain_type_kwargs = {"prompt": prompt}
    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                     chain_type= 'stuff',
                                                     retriever=db.as_retriever(search_kwargs={'k': 1}),
                                                     verbose=True,
                                                     memory=memory,
                                                     combine_docs_chain_kwargs=chain_type_kwargs
                                                     )
    return qa_chain

def init_pinecone():
    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', '700dbf29-7b1d-435b-9da1-c242f7a206e6')
    PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV', 'us-west1-gcp-free')

    pinecone.init( 
        api_key=PINECONE_API_KEY,  
        environment=PINECONE_API_ENV,  
    )
    index_name = "llama2-pdf-chatbox" 
    #embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    #embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    docsearch = Pinecone.from_existing_index(index_name, embeddings)

    return docsearch

def activate_api():
    together.Models.start("togethercomputer/llama-2-7b-chat")

def deactivate_api():
    together.Models.stop("togethercomputer/llama-2-7b-chat")
    
def result():
    # load db from disk
    db = init_pinecone()
    prompt = set_custom_prompt()
    memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    qa = conversationalretrieval_qa_chain(llm, prompt, db, memory)
    #result = qa({"question": ask})
    #res = result['answer']
    
    return qa
    #return res

#chainlit code
@cl.on_chat_start
async def start():
    chain = result()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to X-Fab Hotline Bot. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message, callbacks=[cb])
    
    db = init_pinecone()
    docs = db.similarity_search(message, k=1)

    print(res)
    print(docs)
    
    answer = res["answer"]
    # sources = res["source_documents"]

    # if sources:
    #    answer += f"\nSources:" + str(sources)
    # else:
    #    answer += "\nNo sources found"

    await cl.Message(content=answer).send()


# if __name__ == "__main__":
#     #create_vector_db()
#     while True:
#         query = input(f"\n\nPrompt: " )
#         if query == "exit":
#             print("exiting")
#             break
#         if query == "":
#             continue
#         if query =="start":
#             #activate_api()
#             pass
#         if query == "stop":
#             deactivate_api()
            
#         answer = main(query)  
#         print(answer)
