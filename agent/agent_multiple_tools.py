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
    temperature=0,
    max_tokens=512
)

## This is declaration of multiple Tool

## For duckduckgosearch
search = DuckDuckGoSearchRun()

## For Wikipedia
wikipedia = WikipediaAPIWrapper()

wikipedia_tool = Tool(
    name="wikipedia",
    func=wikipedia.run,
    description="Useful for when you need to look up a topic, country or person on wikipedia."
)

duckduckgo_tool = Tool(
    name="DuckDuckGo Search",
    func=search.run,
    description="Useful for when you need to do a search on the internet to find information that another tool can't find. Be specific with your input."
)

tools = [wikipedia_tool, duckduckgo_tool]

## Initialize the agent
agent = initialize_agent(
    agent="zero-shot-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
    max_iteration=3,
    handle_parsing_errors="Check your output and make sure it conforms!",
)

# print(
#     agent.run(
#         f"Write me an arduino code that read the ultrasonic sensor ?"
#     )
# )

print(
    agent.run(
        "What time is it now in Malaysia?"
    )
)
