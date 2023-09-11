from langchain import PromptTemplate
from langchain import LLMChain
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory

import textwrap

"""
Here, the instructions is part of the system prompt
Here the AI will remember previous conversation history due to `ConversationoBufferMemory

Here we make it with METAL=1
CT_METAL=1 pip install ctransformers --no-binary ctransformers --force-reinstall --upgrade --no-cache-dir --verbose

"""

template = """[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer the question as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
Please answer the question and read the chat history to understand the context
{history}

{question}

<</SYS>>

[/INST]"""


def init_memory():
    memory = ConversationBufferMemory(
        input_key="question",
        memory_key="history",
    )
    return memory

print(template)

MODEL_PATH = r"/Users/jlukas/Desktop/My_Project/NLP/Llama2_quantized_models/llama-2-7b-chat.Q4_K_M.gguf"

prompt = PromptTemplate(input_variables=["history", "question"], template=template)
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
        
        wrapped_text = textwrap.fill(res, width=100)
        print(wrapped_text + "\n\n")
        
        # for word in res:
        #         print(word, end='')
