from langchain import PromptTemplate
from langchain import LLMChain
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory
import textwrap

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.llms import LlamaCpp

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

# GPU Enabled
# llm = CTransformers(model=MODEL_PATH,
#                     model_type='llama',
#                     config={'max_new_tokens': 2048,
#                             'temperature': 0.7,
#                             'gpu_layers': 35,
#                             'stream' : True,
#                             },
                    
#                    )

# CPU Enabled
llm = CTransformers(model=MODEL_PATH,
                    model_type='llama',
                    config={'max_new_tokens': 5000,
                            'temperature': 1,
                            'stream' : True
                            },
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




