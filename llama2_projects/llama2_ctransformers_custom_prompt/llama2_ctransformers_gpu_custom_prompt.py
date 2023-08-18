from langchain import PromptTemplate
from langchain import LLMChain
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory

import textwrap

"""
Here, the instructions is part of the system prompt
Here the AI will remember previous conversation history due to `ConversationoBufferMemory
"""

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

DEFAULT_SYSTEM_PROMPT="""\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

instruction = "write me skillset list set need for Data Scientis engineer \n\n {text}"
SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS
template = B_INST + SYSTEM_PROMPT + instruction + E_INST

template = """[INST] <<SYS>>
Your name is Kelly, You are helpful assistant, you always open and only answer for the assistant then you stop, read the chat history to get the context.
If you don't know the answer, just say you don't know and submit the request to hotline@xfab.com for further assistance.
<</SYS>>
{chat_history}

{user_input}
[/INST]"""

def init_memory():
    memory = ConversationBufferMemory(
        memory_key="chat_history",
    )
    return memory

print(template)

MODEL_PATH = r"D:/llama2_quantized_models/7B_chat/llama-2-7b-chat.ggmlv3.q5_K_M.bin"

prompt = PromptTemplate(input_variables=["chat_history", "user_input"], template=template)
memory = init_memory()

llm = CTransformers(model=MODEL_PATH,
                    model_type='llama',
                    config={'max_new_tokens': 2048,
                            'temperature': 0.7,
                            'gpu_layers': 35,
                            'stream' : True,
                            },
                    
                   )
LLM_Chain=LLMChain(prompt=prompt, llm=llm,memory=memory, verbose=True)

while True:
        query = input(f"\nPrompt >> " )
        if query == "exit":
                print("exiting")
                break
        if query == "":
                continue
        res = LLM_Chain.run(query)
        
        #wrapped_text = textwrap.fill(res, width=100)
        #print(wrapped_text + "\n\n")
        
        for word in res:
                print(word, end='')
