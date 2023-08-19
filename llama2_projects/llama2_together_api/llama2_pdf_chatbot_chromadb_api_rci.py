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
from langchain.embeddings import HuggingFaceEmbeddings

from togetherllm import TogetherLLM
import together

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

from langchain.schema.output_parser import StrOutputParser

import langchain
langchain.debug = False
from operator import itemgetter

from langchain import PromptTemplate
from langchain.prompts.chat import (
    ChatMessagePromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

template = "You are a helpful assistant that imparts wisdom and guide people with accurate answers."

system_message_prompt=SystemMessagePromptTemplate.from_template(template)

human_template="{question}"
human_message_prompt=HumanMessagePromptTemplate.from_template(human_template)

chat_prompt=ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])


#embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
#                                                      model_kwargs={"device": "cuda"})
#embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

PATH = r"C:/Users/Lukas/Desktop/My_Projects/To_Upload/Llama2/llama2_projects/llama2_together_api/data/"
DB_PATH = r"C:/Users/Lukas/Desktop/My_Projects/To_Upload/Llama2/llama2_projects/llama2_together_api/"

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

def load_db():
    db = Chroma(persist_directory=DB_PATH + "/content/db", embedding_function=embeddings)
    return db

def activate_api():
    together.Models.start("togethercomputer/llama-2-7b-chat")

def deactivate_api():
    together.Models.stop("togethercomputer/llama-2-7b-chat")
    
def create_vector_db():
    
    # Load and process the PDF
    loader = DirectoryLoader(PATH, glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(len(documents))
    
    # Split the file into Chunk
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    text = text_splitter.split_documents(documents)
    
    # Create the Database
    # Embed and store the texts
    # Supplying a persist_directory will store the embeddings on disk
    
    persist_directory = DB_PATH + '/content/db'
    db = Chroma.from_documents(text, embeddings, persist_directory=persist_directory)

def recursive(ask,llm,db,memory,prompt):
    template = "You are a helpful assistant that imparts wisdom and guide people with accurate answers."
    system_message_prompt=SystemMessagePromptTemplate.from_template(template)
    human_template="{question} {context}"
    human_message_prompt=HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt=ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    
    qa = conversationalretrieval_qa_chain(llm, prompt, db, memory) 
    result = qa({"question": ask})
    initial_answer = result['answer']
    print(f"Recursive >> {initial_answer}")
    
    return initial_answer
    
def critique(initial_answer,initial_question):
    template = "You are a helpful assistant that looks at answer and finds what is wrong with them based on the original question given"
    system_message_prompt=SystemMessagePromptTemplate.from_template(template)
    human_template="### Question:\n\n{question}\n\n ### Answer Given:{initial_answer}\n\n Review your previous answer and find problems with it"
    human_message_prompt=HumanMessagePromptTemplate.from_template(human_template)
    rc_prompt=ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    
    chain2 = rc_prompt | llm | StrOutputParser()
    constructive_criticsm = chain2.invoke({"question": initial_question, "initial_answer":initial_answer})
    print(f"Critique >> {constructive_criticsm}")
    
    return constructive_criticsm

def improvement(initial_answer,constructive_criticsm,initial_question):
    template = "You are a helpful assistant that looks at answer and finds what is wrong with them based on the original question given"
    system_message_prompt=SystemMessagePromptTemplate.from_template(template)
    human_template="### Question:\n\n{question}\n\n ### Answer Given:{initial_answer}\n\n \
        ###Constructive Criticsm:{constructive_criticsm}\n\n Based on the problem you found, improve your answer.\n\n"
    
    human_message_prompt=HumanMessagePromptTemplate.from_template(human_template)
    improvement_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    
    chain3 = improvement_prompt | llm | StrOutputParser()

    final_result = chain3.invoke({"question": initial_question,
           "initial_answer": initial_answer,
           "constructive_criticsm": constructive_criticsm})
    print(f"Improvement >> {final_result}")
    
    return final_result
    
def main(ask):
    # load db from disk
    db = load_db()
    prompt = set_custom_prompt()
    memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    initial_answer = recursive(ask,llm,db,memory,prompt)
    constructive_criticsm = critique(initial_answer,ask)
    chain3 = improvement(initial_answer,constructive_criticsm,ask) 
    
    return chain3

if __name__ == "__main__":
    #create_vector_db()
    #db = load_db()
    while True:
        query = input(f"\n\nPrompt: " )
        if query == "exit":
            print("exiting")
            break
        if query == "":
            continue
        if query =="start":
            activate_api()
        if query == "stop":
            deactivate_api()
        
        # Answer
        answer = main(query)  
        for word in answer:
            print(word, end='')
        
        # Source and Page
        # sc = db.similarity_search(query,k=1)
        # for i in range(len(sc)):
        #     document = sc[i].metadata.get("source")
        #     page =sc[i].metadata.get("page")
            
        # print(answer)
        # print(f"\n\nSource: " + str(document) + "\nPage: "  + str(page))
