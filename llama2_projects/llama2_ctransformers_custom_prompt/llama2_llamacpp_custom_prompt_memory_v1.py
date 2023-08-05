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
"""

#MODEL_PATH = r"D:/llama2_quantized_models/7B_chat/llama-2-7b-chat.ggmlv3.q8_0.bin"
#MODEL_PATH = r"D:/llama2_quantized_models/7B_chat/llama-2-7b-chat.ggmlv3.q4_K_M.bin"
MODEL_PATH = r"D:/llama2_quantized_models/7B_chat/llama-2-7b-chat.ggmlv3.q5_K_M.bin"

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

CUSTOM_SYSTEM_PROMPT = """You are helpful assistant, you always only answer for the assistant then you stop, read the chat history to get the context"""
INSTRUCTION = "Chat History:\n\n{chat_history} \n User:\n {user_input}"
SYSTEM_PROMPT = B_SYS + CUSTOM_SYSTEM_PROMPT + INSTRUCTION + E_SYS
template = B_INST + SYSTEM_PROMPT + E_INST
print(template)

def init_memory():
    memory = ConversationBufferMemory(
        input_key="user_input",
        memory_key="chat_history",
    )
    return memory

prompt = PromptTemplate(input_variables=["chat_history", "user_input"], template=template)
memory = init_memory()

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




