"""
Agent will perform this 3 things
1. observation
2. thought
3. action
"""

from togetherllm import TogetherLLM
from langchain.utilities import WikipediaAPIWrapper
from langchain.utilities import PythonREPL
from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.utilities import SerpAPIWrapper

from langchain.agents import Tool, AgentExecutor, initialize_agent, load_tools
from langchain.agents import AgentType

from pprint import pprint
import os

os.environ["SERPAPI_API_KEY"] = "f078f92ace79931f581dcb74e38817617d839c30aed3bd08c834ac46c1942ff2"

llm = TogetherLLM(
    model= "togethercomputer/llama-2-7b-chat",
    temperature=0,
    max_tokens=512
)

## This is declaration of single Tool using Load Tools

tools = load_tools(["serpapi"], llm=llm)

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
        f"what time is it now at kuching?"
    )
)