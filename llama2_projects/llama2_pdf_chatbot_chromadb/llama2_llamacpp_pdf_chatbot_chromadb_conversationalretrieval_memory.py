
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader

from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp
import chromadb

Path = r"D:/AI_CTS/Llama2/llama2_projects/llama2_pdf_chatbot_chromadb/"
MODEL_PATH = r"D:/AI_CTS/Llama2/llama2_projects/llama2_quantized_models/7B_chat/llama-2-7b-chat.ggmlv3.q5_K_M.bin"

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT="""\
You are a helpful, respectful and honest assistant. Always answer the question as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

instruction = "Given the context that has been provided. \n {context}, Answer the following question -\n{question}"
system_prompt ="""You are an expert technical chat support.
You will be a given a context to answer from. Be precise in your answers wherever possible.
In case you are sure you don't know the answer then you say that based on the context that you don't know the answer.
In all other instnaces you provide an answer to the best of your capability. Cite url when you can access them related to the context."""

def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST

    print(prompt_template)

    return prompt_template

def chunking():
    loader = PyPDFLoader(r"D:/AI_CTS/Llama2/llama2_projects/llama2_pdf_chatbot_chromadb/data/Hotline_Wiki.pdf")

    tex_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 20,
        length_function = len,
    )
    pages = loader.load_and_split(tex_splitter)
    return pages

def vector_storing(pages):
    # Save db to disk
    db = Chroma.from_documents(pages, HuggingFaceEmbeddings(), persist_directory=Path + "/content/db")
    
    retriever = db.as_retriever()
    return retriever

def init_model():
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(
        model_path= MODEL_PATH,
        max_tokens=256,
        n_gpu_layers=32,
        n_batch= 512, #256,
        callback_manager=callback_manager,
        n_ctx= 1024,
        verbose=False,
        temperature=0.8,
    )
    return llm

def Conversational_Retrieveal(llm,db,memory,prompt):
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 3}),
        verbose=False,
        memory = memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    return qa

def ingest():
    doc = chunking()
    db = vector_storing(doc)

def main(query):
    # load db from disk
    db = Chroma(persist_directory=Path + "/content/db", embedding_function=HuggingFaceEmbeddings())
    
    #search = db.similarity_search("what is FLATPV?")
    #print(search)

    template = get_prompt(instruction,system_prompt)
    prompt = PromptTemplate(template=template, input_variables=["context","question"])
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history", k=5,
        return_messages=True
    )
    llm = init_model()
    qa = Conversational_Retrieveal(llm,db,memory,prompt)

    result = qa({"question": query})
    res = result['answer']

    return res

def final_result(query):
    response = main(query)
    return response

if __name__ == "__main__":
    #ingest()
    #main()
    while True:
        query = input(f"\n\nPrompt: " )
        if query == "exit":
            print("exiting")
            break
        if query == "":
            continue
        answer = final_result(query)  
    