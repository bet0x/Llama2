from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain import PromptTemplate, LLMChain
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

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
from pprint import pprint

llm = TogetherLLM(
    model= "togethercomputer/llama-2-7b-chat",
    temperature=0,
    max_tokens=512
)

def recursive():
    template = "You are a helpful assistant that imparts wisdom and guide people with accurate answers."
    system_message_prompt=SystemMessagePromptTemplate.from_template(template)
    human_template="{question}"
    human_message_prompt=HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt=ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    
    chain1 = chat_prompt | llm | StrOutputParser()
    
    return chain1
    
def critique(chain1):
    template = "You are a helpful assistant that looks at answer and finds what is wrong with them based on the original question given"
    system_message_prompt=SystemMessagePromptTemplate.from_template(template)
    human_template="### Question:\n\n{question}\n\n ### Answer Given:{initial_answer}\n\n Review your previous answer and find problems with it"
    human_message_prompt=HumanMessagePromptTemplate.from_template(human_template)
    rc_prompt=ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    
    critique_chain = {"question": itemgetter("question"),
                  "initial_answer": chain1} | rc_prompt | llm | StrOutputParser()
    
    return critique_chain

def improvement(critique_chain,chain1,question):
    template = "You are a helpful assistant that looks at answer and finds what is wrong with them based on the original question given"
    system_message_prompt=SystemMessagePromptTemplate.from_template(template)

    human_template="### Question:\n\n{question}\n\n ### Answer Given:{initial_answer}\n\n \
    ###Constructive Criticsm:{constructive_criticsm}\n\n Based on the problem you found, improve your answer.\n\n"
    
    human_message_prompt=HumanMessagePromptTemplate.from_template(human_template)
    improvement_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    
    chain3 = {"question": itemgetter("question"),
           "initial_answer": chain1,
           "constructive_criticsm": critique_chain} | improvement_prompt | llm | StrOutputParser()
    
    res = chain3.invoke({"question": {question}})
    for word in res:
        print(word, end='')

if __name__ == "__main__":
    while True:
        query = input(f"\n\nPrompt: " )
        if query == "exit":
            print("exiting")
            break
        if query == "":
            continue
        chain1 = recursive()
        chain2 = critique(chain1)
        chain3 = improvement(chain2,chain1,query)
