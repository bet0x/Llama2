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
import gradio as gr
import random
import time

from togetherllm import TogetherLLM

Path = r"D:/AI_CTS/Llama2/llama2_projects/llama2_pdf_chatbot_chromadb/"
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

memory = ConversationBufferWindowMemory(memory_key="chat_history", k=5,return_messages=True)
db = Chroma(persist_directory=Path + "/content/db", embedding_function=HuggingFaceEmbeddings())

llm = TogetherLLM(
    model= "togethercomputer/llama-2-7b-chat",
    temperature=0,
    max_tokens=512
)
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

def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

with gr.Blocks() as demo:
    
    template = get_prompt(instruction,system_prompt)
    prompt = PromptTemplate(template=template, input_variables=["context","question"])
    qa = Conversational_Retrieveal(llm,db,memory,prompt)

    update_sys_prompt = gr.Textbox(label="update system prompt")
    chatbot = gr.Chatbot(label="Chess Bot", height=600)
 
    msg = gr.Textbox(label="Question")
    clear=gr.ClearButton([msg,chatbot])
    clear_memory = gr.Button(value = "Clear LLM Memory")
    
    def respond(message,chat_history):
        bot_message = qa({"question": message})["answer"]
        chat_history.append([message,bot_message])
        return "",chat_history
    
    def update_prompt(sys_prompt):
        if sys_prompt == "":
            sys_prompt = system_prompt
        template = get_prompt(instruction,sys_prompt)
        prompt = PromptTemplate(template=template, input_variables=["context","question"])
        qa.combine_docs_chain.llm_chain.prompt = prompt

    def clear_llm_memory():
        memory.clear()

    msg.submit(respond, inputs=[msg,chatbot], outputs=[msg,chatbot])
    clear_memory.click(clear_llm_memory)
    update_sys_prompt.submit(update_prompt,inputs=[update_sys_prompt])

demo.launch(share=False, debug=True)