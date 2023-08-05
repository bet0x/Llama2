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

template = """[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer the question as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.

{chat_history}

{user_input}
<</SYS>>

[/INST]"""

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




