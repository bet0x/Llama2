from togetherllm import TogetherLLM
from langchain.utilities import WikipediaAPIWrapper
from langchain.utilities import PythonREPL
from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.utilities import SerpAPIWrapper

from langchain.agents import Tool, AgentExecutor, initialize_agent
from langchain.agents import AgentType

from pprint import pprint
import os

llm = TogetherLLM(
    model= "togethercomputer/llama-2-7b-chat",
    temperature=0.7,
    max_tokens=512
)

## This is declaration of single Tool
## For duckduckgosearch
search = DuckDuckGoSearchRun()

tools = [
    Tool(
        name="DuckDuckGo Search",
        func=search.run,
        description="Useful for when you need to do a search on the internet to find information that another tool can't find. Be specific with your input."
    )
]

## Initialize the agent
agent = initialize_agent(
    agent="zero-shot-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
    max_iteration=3,
    handle_parsing_errors="Check your output and make sure it conforms!",
)

print(
    agent.run(
        f"Write me an arduino code that turn off and on all the five leds ?"
    )
)