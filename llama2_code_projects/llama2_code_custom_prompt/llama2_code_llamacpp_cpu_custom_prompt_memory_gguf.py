import gradio as gr
from langchain import PromptTemplate
from langchain import LLMChain
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory
import textwrap

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.llms import LlamaCpp
from time import sleep

# GGUF
MODEL_PATH = r"D:/AI_CTS/Llama2/llama2_code_projects/llama2_code_quantized_models/7B_Instructional/codellama-7b-instruct.Q2_K.gguf"

custom_prompt_template = """[INST] <<SYS>>
You are an AI Coding Assistant and your task is to solve coding problems and return code snippets based on given user's query. Below is the user's query.
<</SYS>>
{chat_history}

Query: {query}
You just return the helpful code.
Helpful Answer:
[/INST]"""

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
prompt = PromptTemplate(input_variables=["chat_history","query"], template=custom_prompt_template)
memory = ConversationBufferMemory(input_key="query",memory_key="chat_history",)

llm = LlamaCpp(
    model_path= MODEL_PATH,
    max_tokens=256,
    n_gpu_layers=32, #32, # for M1 set equal to 1 only # for ubuntu or windows set depend on layer = 32
    n_ctx= 2048,
    n_batch= 512, #256,
    callback_manager=callback_manager,
    top_p=1,
    verbose=True,
    temperature=0.8,
    f16_kv = True
)
print(llm.n_gpu_layers)


def chain_pipeline():
    qa_prompt = prompt
    qa_chain = LLMChain(prompt=qa_prompt, memory=memory, llm=llm)
    return qa_chain

llmchain = chain_pipeline()

def bot(query):
    llm_response = llmchain.run({"query": query})
    return llm_response

with gr.Blocks(title='Code Llama Demo') as demo:
    gr.Markdown("# Code Llama Demo")
    chatbot = gr.Chatbot([], elem_id="chatbot", height=700)
    msg = gr.Textbox()
    clear = gr.ClearButton([msg,chatbot])

    def respond(message, chat_history):
        bot_message = bot(message)
        chat_history.append((message, bot_message))
        sleep(2)
        return "", chat_history
    
    msg.submit(respond, [msg, chatbot], [msg, chatbot])

demo.launch()