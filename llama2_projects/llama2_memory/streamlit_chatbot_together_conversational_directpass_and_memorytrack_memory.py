import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Pinecone
import pinecone
import os
from togetherllm import TogetherLLM
from langchain import PromptTemplate
from time import sleep

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

chat_history = []

llm = TogetherLLM(
model= "togethercomputer/llama-2-7b-chat",
temperature=0,
max_tokens=512
)

custom_prompt_template = """[INST] <<SYS>>
Your name is Kelly, you are a helpful, respectful and honest assistant. Always answer as helpfully as possible using the context text provided.
You answer should only answer the question once and not have any text after the answer is done.\n\nIf a question does not make any sense, or is not factually
coherent, explain why instead of answering something not correct. If you don't know the answer, just say you don't know and submit the request to hotline@xfab.com for further assistance.\n
<</SYS>>
CONTEXT:/n/n {context}/n
{chat_history}
Question: {question}
[/INST]"""

print(custom_prompt_template)

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(input_variables=['chat_history','context', 'question'],
                            template=custom_prompt_template)
    return prompt

def init_pinecone():
    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', '700dbf29-7b1d-435b-9da1-c242f7a206e6')
    PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV', 'us-west1-gcp-free')

    pinecone.init( 
        api_key=PINECONE_API_KEY,  
        environment=PINECONE_API_ENV,  
    )
    index_name = "new-wikidb-v1" 
    docsearch = Pinecone.from_existing_index(index_name, embeddings)

    return docsearch

def conversationalretrieval_qa_chain(llm,qa_prompt,db):
    chain_type_kwargs = {"prompt": qa_prompt}
    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm,chain_type='stuff',
                                                retriever=db.as_retriever(search_kwargs={"k":2}),
                                                #verbose=True,
                                                memory=memory,
                                                combine_docs_chain_kwargs=chain_type_kwargs)

    return qa_chain

def conversation_chat(query):
    db = init_pinecone()
    qa_prompt = set_custom_prompt()
    qa = conversationalretrieval_qa_chain(llm,qa_prompt,db)
    
    # Memory Context
    result = qa({"question": query})

    # Direct Pass
    #result = qa({"question": query, "chat_history": chat_history})
    
    print('Answer: ' + result['answer'])
    chat_history.append((query, result['answer']))

    #for word in answer:
    #        print(word, end='')
    #return result["answer"]


if __name__ == "__main__":
    while True:
        query = input(f"\n\nPrompt: " )
        if query == "exit":
            print("exiting")
            break
        if query == "":
            continue

        # Answer
        answer = conversation_chat(query)  
       

