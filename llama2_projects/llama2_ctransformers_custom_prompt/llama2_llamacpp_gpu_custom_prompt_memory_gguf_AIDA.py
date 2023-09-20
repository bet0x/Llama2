from langchain import PromptTemplate
from langchain import LLMChain
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory
import textwrap

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.llms import LlamaCpp

# '''
# Here, the instructions is part of the system prompt
# Here the AI will remember previous conversation history due to `ConversationoBufferMemory

# Here we make it with CPU Support
# Without CUDA support

# OpenBLAS:
# This provides BLAS acceleration using only the CPU. Make sure to have OpenBLAS installed on your machine.

# 1. Download the latest fortran version of w64devkit (https://github.com/skeeto/w64devkit/releases)

# 2. Download the latest version of OpenBLAS for Windows (https://github.com/xianyi/OpenBLAS/releases)

# 3. Extract w64devkit on your pc.

# 4. From the OpenBLAS zip that you just downloaded copy libopenblas.a, located inside the lib folder, inside w64devkit\x86_64-w64-mingw32\lib.

# 5. From the same OpenBLAS zip copy the content of the include folder inside w64devkit\x86_64-w64-mingw32\include.

# 6. Run w64devkit.exe.

# Use the cd command to reach the llama.cpp folder.

# From here you can run "make LLAMA_OPENBLAS=1"

# To set an environment:
# (LLAMA) PS C:\Users\jlukas> $Env:CMAKE_ARGS="-DLLAMA_BLAS=on -DLLAMA_BLAS_VENDOR=openBLAS"  
# (LLAMA) PS C:\Users\jlukas> $Env:FORCE_CMAKE=1    

# pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir --verbose

# Download GGUF model
# https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/tree/main

# Please remember that latest release of llamaCpp is no longer support GGML. We will need to use GGUF model format.
# '''

## GGML
#MODEL_PATH = r"D:/llama2_quantized_models/7B_chat/llama-2-7b-chat.ggmlv3.q4_K_M.bin"

# GGUF
MODEL_PATH = r"/home/jlukas/Desktop/My_Project/AI_CTS/Llama2_Quantized/7B_GGUF/llama-2-7b-chat.Q4_K_M.gguf"

template = """[INST] <<SYS>>
Your name is Kelly.
You are helpful assistant, you always only answer for the question then you stop, read the chat history to get the context
You always answer the question as helpfully as possible.
<</SYS>>
{chat_history}

Question: {question}
[/INST]"""

print(template)

def init_memory():
    memory = ConversationBufferMemory(
        input_key="question",
        memory_key="chat_history",
    )
    return memory

prompt = PromptTemplate(input_variables=["chat_history", "question"], template=template)
memory = init_memory()

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# GPU Enabled
# llm = LlamaCpp(
#     model_path= MODEL_PATH,
#     max_tokens=256,

#     n_gpu_layers=32,
#     n_batch= 512, #256,

#     callback_manager=callback_manager, 
#     n_ctx= 1024, 
#     #verbose=True, 
#     temperature=0.8, 
# )
# print(llm.n_gpu_layers)

# CPU Enabled
llm = LlamaCpp(
    model_path= MODEL_PATH,
    max_tokens=5000,#2000,
    n_ctx= 5000, #2048,
    temperature=0.75,
    f16_kv = True, # Must set to True, otherwise you will run into a problem after couple of calss
    top_p=1,
    callback_manager=callback_manager, 
    n_gpu_layers=100,
    verbose=True, # Verbose is required to pass to the callback manager
)

LLM_Chain=LLMChain(prompt=prompt, memory=memory, llm=llm)

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




