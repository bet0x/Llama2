from langchain import PromptTemplate
from langchain import LLMChain
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory
import textwrap

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.llms import LlamaCpp

# GGUF
MODEL_PATH = r"D:/AI_CTS/Llama2/llama2_code_projects/llama2_code_quantized_models/7B_Instructional/codellama-7b-instruct.Q4_K_M.gguf"

custom_prompt_template = """[INST] <<SYS>>
You are an AI Coding Assitant and your task is to solve coding problems and return code snippets based on given user's query. Below is the user's query.
<</SYS>>
Query: {question}

You just return the helpful code.
Helpful Answer:
[/INST]"""

# custom_prompt_template = """[INST] <<SYS>>
# You are a helpful assistant, you will use the provided context to answer user questions.
# Read the given context before answering questions and think step by step. If you can  not answer a user question based on
# the provided context, inform the user. Do not use any other information for answering user
# <</SYS>>

# User: {question}
# [/INST]"""

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

prompt = PromptTemplate(input_variables=["question"], template=custom_prompt_template)

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

LLM_Chain=LLMChain(prompt=prompt, llm=llm)

while True:
        query = input(f"\nPrompt >> " )
        if query == "exit":
                print("exiting")
                break
        if query == "":
                continue
        res = LLM_Chain.run(query)
        wrapped_text = textwrap.fill(res, width=100)
        print(wrapped_text + "\n\n")