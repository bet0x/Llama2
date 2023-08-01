import streamlit as st
from streamlit_chat import message
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

# use for GPU
#from ctransformers import AutoModelForCausalLM

# Use for CPU
from langchain.llms import CTransformers

PATH = r"D:/AI_CTS/Llama2/llama2_projects/llama2_url_chatbot/"

#MODEL_PATH = r"D:/AI_CTS/Llama2/llama2_projects/llama2_quantized_models/3B_Orca/"
MODEL_PATH = r"D:/AI_CTS/Llama2/llama2_projects/llama2_quantized_models/7B_chat/"

DB_FAISS_PATH = PATH + 'vectorstore/db_faiss'

st.title("Hotline Virtual Assistant👩‍💼")
st.markdown("<h3 style='text-align: center; color: white;'>Built by <a href='https://github.com/AIAnytime'>X-Fab with ❤️ </a></h3>", unsafe_allow_html=True)

#Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = MODEL_PATH + "llama-2-7b-chat.ggmlv3.q8_0.bin",
        #model = MODEL_PATH + "orca-mini-3b.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
    )

    # With GPU 
    #llm = AutoModelForCausalLM.from_pretrained(MODEL_PATH + "llama-2-7b-chat.ggmlv3.q8_0.bin", gpu_layers=50)

    return llm

def conversational_chat(query):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                    model_kwargs={'device': 'cpu'})
db = FAISS.load_local(DB_FAISS_PATH, embeddings)
llm = load_llm()
chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever(search_kwargs={'k': 2}))


if 'history' not in st.session_state:
    st.session_state['history'] = []

# Output
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello ! Ask me about anything " + "😊"]
    
# User Input
if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey ! 👋"]

#container for the chat history
response_container = st.container()

#container for the user's text input
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        
        user_input = st.text_input("Query:", placeholder="Search your data here :", key='input')
        submit_button = st.form_submit_button(label='Send')
        
    if submit_button and user_input:
        output = conversational_chat(user_input)
        
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
            message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")


