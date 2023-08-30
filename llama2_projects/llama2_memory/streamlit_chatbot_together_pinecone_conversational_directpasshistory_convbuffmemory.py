'''
Pass in chat history 
We can also just pass it in explicitly. In order to do this, we need to initialize
a chain without any memory object.
'''

import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
import os
from togetherllm import TogetherLLM
from langchain import PromptTemplate
from time import sleep

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
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

#print(custom_prompt_template)

# Set Streamlit page configuration
# Initialize session states
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if 'history' not in st.session_state:
    st.session_state['history'] = []

with st.sidebar:
    st.title("X-Chat")
    st.header("Settings")
    add_replicate_api = st.text_input("Enter your password here", type='password')

    if not (add_replicate_api.startswith('jl')):
        st.warning('Please enter your credentials', icon='⚠️')
    else:
        st.success('Proceed to entering your prompt message!', icon='👉')

if prompt := st.chat_input(st.session_state["input"], disabled=not add_replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]
    
# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

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
                                                verbose=True,
                                                combine_docs_chain_kwargs=chain_type_kwargs)

    return qa_chain

def conversation_chat(query):
    db = init_pinecone()
    qa_prompt = set_custom_prompt()
    qa = conversationalretrieval_qa_chain(llm,qa_prompt,db)

    # Here we pass the chat history directly wihout using memory buffer
    result = qa({"question": query, "chat_history": st.session_state['history']})

    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

#Clear the chat messages
def clear_chat_history():
   st.session_state.messages=[{"role": "assistant","content": "How may i assist you today ?"}]

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking"):
            response = conversation_chat(prompt)
            
            st.session_state.past.append(prompt)
            st.session_state.generated.append(response)
            placeholder = st.empty()
            full_response=''
            for item in response:
                full_response+=item
                sleep(0.01)
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
            with st.expander("Conversation"):
                for i in range(len(st.session_state['generated'])-1, -1, -1):
                    st.info(st.session_state["past"][i],icon="🧐")
                    st.success(st.session_state["generated"][i], icon="🤖")
                    st.info(st.session_state["history"][i],icon="📚")                
                #st.markdown(st.session_state.generated)
                    
    message = {"role":"assistant", "content": full_response}
    st.session_state.messages.append(message)