import streamlit as st
from st_pages import Page, Section, show_pages, add_page_title
from st_pages import show_pages_from_config
import pandas as pd
import numpy as np
import random

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

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

st.set_page_config(
    page_title="Wikipedia",
     page_icon="https://api.dicebear.com/5.x/bottts-neutral/svg?seed=gptLAb"#,
)

add_page_title()

st.markdown("""### <span style="color:green"> In this new revise architecture, we're using RAG + LLM approach</span>""",unsafe_allow_html=True)
st.markdown("""
            * Initial question and Output data from retrieval will be fed as an input to LLM model.
            * LLM model will generate human response based on information.

            #### I'm using below prompt to get coincise and direct response from the model 
            """)
st.code("""
    [INST] <<SYS>>
    Your name is Kelly, you are a helpful, respectful and honest assistant. Always answer as helpfully as possible using the context text provided.
    You answer should only answer the question once and not have any text after the answer is done.

    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the
    answer, just say you don't know and submit the request to hotline@xfab.com for further assistance.

    <</SYS>>
    CONTEXT:/n/n {context}/n
    Question: {question}
    [/INST]
""")

custom_prompt_template = """[INST] <<SYS>>
Your name is Kelly, you are a helpful, respectful and honest assistant. Always answer as helpfully as possible using the context text provided.
You answer should only answer the question once and not have any text after the answer is done and please ensure your answer is relevant to the question.\n\nIf a question does not make any sense, or is not factually
coherent, explain why instead of answering something not correct. If you don't know the answer, just say you don't know and submit the request to hotline@xfab.com for further assistance.\n
<</SYS>>
CONTEXT:/n/n {context}/n
{chat_history}
Question: {question}
[/INST]"""

print(custom_prompt_template)

st.title("Wiki 3.0 Decoder Encoder")
prompt=st.text_input("Enter your question here")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

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
                                                     retriever=db.as_retriever(search_kwargs={'k': 1}),
                                                     verbose=True,
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
        index_name="new_wikidb_v1",
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

if prompt:
    with st.expander("Return result"):
        answer = qa_bot(prompt)
        st.info(answer)
        print(answer)
        
with st.expander("Sample Output"):
    st.image("./images/wiki_3_0_image.jpg", caption="Fig 1: LLM + RAG result")
# while True:
#     query = input(f"\n\nPrompt: " )
#     if query == "exit":
#         print("exiting")
#         break
#     if query == "":
#         continue
#     answer = qa_bot(query)
#     print(answer)
    

# ###################################
# df = pd.DataFrame(
#    np.random.randn(50, 20),
#    columns=('col %d' % i for i in range(20)))

# st.dataframe(df)  # Same as st.write(df)

# df = pd.DataFrame(
#    np.random.randn(10, 20),
#    columns=('col %d' % i for i in range(20)))

# st.dataframe(df.style.highlight_max(axis=0))

# ####################################
# df = pd.DataFrame(
#     {
#         "name": ["Roadmap", "Extras", "Issues"],
#         "url": ["https://roadmap.streamlit.app", "https://extras.streamlit.app", "https://issues.streamlit.app"],
#         "stars": [random.randint(0, 1000) for _ in range(3)],
#         "views_history": [[random.randint(0, 5000) for _ in range(30)] for _ in range(3)],
#     }
# )

# st.dataframe(
#     df,
#     column_config={
#         "name": "App name",
#         "stars": st.column_config.NumberColumn(
#             "Github Stars",
#             help="Number of stars on GitHub",
#             format="%d ⭐",
#         ),
#         "url": st.column_config.LinkColumn("App URL"),
#         "views_history": st.column_config.LineChartColumn(
#             "Views (past 30 days)", y_min=0, y_max=5000
#         ),
#     },
#     hide_index=True,
# )

# ##################################
# # Cache the dataframe so it's only loaded once
# @st.cache_data
# def load_data():
#     return pd.DataFrame(
#         {
#             "first column": [1, 2, 3, 4],
#             "second column": [10, 20, 30, 40],
#         }
#     )

# # Boolean to resize the dataframe, stored as a session state variable
# st.checkbox("Use container width", value=False, key="use_container_width")

# df = load_data()

# # Display the dataframe and allow the user to stretch the dataframe
# # across the full width of the container, based on the checkbox value
# st.dataframe(df, use_container_width=st.session_state.use_container_width)

# ################################

# import pandas as pd
# import streamlit as st

# data_df = pd.DataFrame(
#     {
#         "widgets": ["st.selectbox", "st.number_input", "st.text_area", "st.button"],
#     }
# )

# st.data_editor(
#     data_df,
#     column_config={
#         "widgets": st.column_config.Column(
#             "Streamlit Widgets",
#             help="Streamlit **widget** commands 🎈",
#             width="medium",
#             required=True,
#         )
#     },
#     hide_index=True,
#     num_rows="dynamic",
# )