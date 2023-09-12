from langchain import PromptTemplate
from langchain import LLMChain
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory
import textwrap

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.llms import LlamaCpp

"""
Here, the instructions is part of the system prompt
Here the AI will remember previous conversation history due to `ConversationoBufferMemory

Here we make it with METAL=1
CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir --verbose

Download GGUF model
https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/tree/main

Please remember that latest release of llamaCpp is no longer support GGML. We will need to use GGUF model format.
"""

## GGML
#MODEL_PATH = r"D:/llama2_quantized_models/7B_chat/llama-2-7b-chat.ggmlv3.q8_0.bin"
#MODEL_PATH = r"D:/llama2_quantized_models/7B_chat/llama-2-7b-chat.ggmlv3.q4_K_M.bin"
#MODEL_PATH = r"D:/llama2_quantized_models/7B_chat/llama-2-7b-chat.ggmlv3.q5_K_M.bin"

# GGUF
MODEL_PATH = r"/Users/jlukas/Desktop/My_Project/NLP/Llama2_quantized_models/llama-2-7b-chat.Q4_K_M.gguf"

template = """[INST] <<SYS>>
Your name is Kelly.
You are helpful assistant, you always only answer for the question then you stop, read the chat history to get the context
You always answer the question as helpfully as possible.
<</SYS>>
{chat_history}

Question: {user_input}
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

llm = LlamaCpp(
    model_path= MODEL_PATH,
    max_tokens=256,
    n_gpu_layers=32, # for M1 set equal to 1 only # for ubuntu or windows set depend on layer = 32
    n_batch= 512, #256,
    callback_manager=callback_manager,
    #verbose=True,
    temperature=0.8,
)
print(llm.n_gpu_layers)

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




