import gradio as gr
from langchain import PromptTemplate
from langchain import LLMChain
from langchain.memory import ConversationBufferMemory

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.llms import LlamaCpp

# GGUF
MODEL_PATH = r"D:/AI_CTS/Llama2/llama2_code_projects/llama2_code_quantized_models/7B_Instructional/codellama-7b-instruct.Q4_K_M.gguf"

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path= MODEL_PATH,
    max_tokens=5000,#2000,
    n_ctx= 5000, #2048,
    temperature=0.75,
    f16_kv = True, # Must set to True, otherwise you will run into a problem after couple of calss
    top_p=1,
    callback_manager=callback_manager, 
    verbose=True, # Verbose is required to pass to the callback manager
)

def llm_function(message,chat_history):
    res = llm(message)
    return res

title = "CodeLlama 7B GGUF Demo"

examples = [
    'Write a python code to connect with a SQL database and list down all the tables.',
    'Write the python code to train a linear regression model using Scikit Learn.',
    'Explain the concepts of Functional Programming.',
    'Can you explain the benefits of Python programming language?'
]

gr.ChatInterface(
    fn=llm,
    title=title,
    examples=examples
).launch()
