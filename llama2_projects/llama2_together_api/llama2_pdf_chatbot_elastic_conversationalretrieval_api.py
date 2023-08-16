'''
Visit Together.ai : https://api.together.xyz/settings/billing
'''

from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate, LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import ElasticVectorSearch

import together
from togetherllm import TogetherLLM

# Use for CPU
#from langchain.llms import CTransformers
#from ctransformers.langchain import CTransformers

# Use for GPU
from langchain.llms import LlamaCpp

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

custom_prompt_template = """[INST] <<SYS>>
Your name is Kelly, You are helpful assistant, you always open and only answer for the assistant then you stop, read the chat history to get the context.
If you don't know the answer, just say you don't know and submit the request to hotline@xfab.com for further assistance.
<</SYS>>
{context}

{chat_history}
Question: {question}
[/INST]"""

print(custom_prompt_template)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                            model_kwargs={'device': 'cpu'})
llm = TogetherLLM(
    model= "togethercomputer/llama-2-7b-chat",
    temperature=0.1,
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
                                                     retriever=db.as_retriever(search_kwargs={'k': 3}),
                                                     verbose=False,
                                                     memory=memory,
                                                     combine_docs_chain_kwargs=chain_type_kwargs
                                                     )
    return qa_chain
    
def load_db():
    CERT_FINGERPRINT = "7e73d3cf8918662a27be6ac5f493bf55bd8af2a95338b9b8c49384650c59db08"
    CERT_PATH = "D:\elasticsearch-8.4.2\config\certs\http_ca.crt"
    ELASTIC_PASSWORD = "Eldernangkai92"
    elasticsearch_url = f"https://elastic:Eldernangkai92@localhost:9200"

    db= ElasticVectorSearch(
        elasticsearch_url=elasticsearch_url,
        index_name="elastic_wiki",
        ssl_verify={
            "verify_certs": True,
            "basic_auth": ("elastic", ELASTIC_PASSWORD),
            "ssl_assert_fingerprint" :  CERT_FINGERPRINT # You can use fingerprint also
            #"ca_certs": CERT_PATH, # You can Certificate path too
        },
        embedding=embeddings
    )
    
    return db

def semantic_search(docsearch,query):
    docs=docsearch.similarity_search(query)
    return docs

#QA Model Function
def qa_bot(ask):

    db = load_db()
    prompt = set_custom_prompt()
    memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    qa = conversationalretrieval_qa_chain(llm, prompt, db, memory)
    
    result = qa({"question": ask})
    res = result['answer']

    #res = llm(ask)
    #print(res)
    #return res

    # llm_chain = LLMChain(
    # llm=llm,
    # prompt=prompt,
    # verbose=True,
    # memory=memory,)

    # res = llm_chain.predict(user_input=ask)
    #print(res)

    return res


while True:
    query = input(f"\n\nPrompt: " )
    if query == "exit":
        print("exiting")
        break
    if query == "":
        continue
    answer = qa_bot(query)
    print(answer)
    
    
