from togetherllm import TogetherLLM
from langchain.utilities import WikipediaAPIWrapper
from langchain.utilities import PythonREPL
from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.utilities import SerpAPIWrapper

from langchain.agents import Tool, AgentExecutor, initialize_agent
from langchain.agents import AgentType
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from pprint import pprint
import os

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

llm = TogetherLLM(
    model= "togethercomputer/llama-2-7b-chat",
    temperature=0,
    max_tokens=512
)
prompt = PromptTemplate(input_variables=["query"],template="{query}")

## For duckduckgosearch
duck = DuckDuckGoSearchRun()

tools = [
    Tool(
        name="DuckDuckGo",
        func=duck.run,
        description="Useful for when you need to do a search on the internet to find information that another tool can't find. Be specific with your input."
    )
]

## For LLM Chain
llm_chain = LLMChain(llm=llm, prompt=prompt)

# initialize the LLM tool
llm_tool = Tool(
    name='Language Model',
    func=llm_chain.run,
    description='use this tool for general purpose queries and logic'
)

tools.append(llm_tool)

zero_shot_agent = initialize_agent(
    agent="zero-shot-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
    handle_parsing_errors="Check your output and make sure it conforms!",
    max_iterations=3,
)

print(
    zero_shot_agent("Write me a romantic letter to my girlfriend?")
)