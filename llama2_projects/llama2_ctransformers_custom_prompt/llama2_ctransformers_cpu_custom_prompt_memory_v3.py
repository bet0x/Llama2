from langchain import PromptTemplate
from langchain import LLMChain
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory
import textwrap

"""
Here, the instructions is not part of the system prompt.
Here the AI will remember previous conversation history due to `ConversationoBufferMemory
"""
MODEL_PATH = r"D:/llama2_quantized_models/7B_chat/llama2.7b.airoboros.ggml_v3.q4_K_M.bin"
#MODEL_PATH = r"D:/llama2_quantized_models/7B_chat/llama-2-7b-chat.ggmlv3.q8_0.bin"

template = """[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer the question as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
Please answer the question and read the chat history to understand the context
{history}

{question}

<</SYS>>

[/INST]"""

print(template)
prompt = PromptTemplate(input_variables=["history","question"], template=template)
memory = ConversationBufferMemory(input_key="question", memory_key="history")

llm = CTransformers(model=MODEL_PATH,
        model_type='llama',
        config={'max_new_tokens': 128,
                'temperature': 0.01}
        )

LLM_Chain=LLMChain(prompt=prompt, llm=llm, memory=memory, verbose=True)

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
