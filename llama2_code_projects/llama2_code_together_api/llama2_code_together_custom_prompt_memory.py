from langchain import PromptTemplate
from langchain import LLMChain
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory
import textwrap

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from togetherllm import TogetherLLM

custom_prompt_template = """[INST] <<SYS>>
You are an AI Coding Assistant and your task is to solve coding problems and return code snippets based on given user's query. Below is the user's query.
<</SYS>>
{chat_history}

Query: {query}
You just return the helpful code.
Helpful Answer:
[/INST]"""

llm = TogetherLLM(
    model= "togethercomputer/CodeLlama-7b-Instruct",
    temperature=0,
    max_tokens=512
)

prompt = PromptTemplate(input_variables=["chat_history","query"], template=custom_prompt_template)
memory = ConversationBufferMemory(input_key="query",memory_key="chat_history",)

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