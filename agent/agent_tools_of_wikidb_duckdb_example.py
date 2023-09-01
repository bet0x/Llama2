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

## For Wikipedia
wikipedia = WikipediaAPIWrapper()

## For PythonREPL
python_repl = PythonREPL()

## For duckduckgosearch
search = DuckDuckGoSearchRun()

## For GoogleSearch
os.environ["GOOGLE_CSE_ID"] = "4491a51aa54194a90"
os.environ["GOOGLE_API_KEY"] = "AIzaSyAoFhZUM4dwrLgbGwFzjukOMjzCLujtjE0"
google = GoogleSearchAPIWrapper()

## For SERPAPI
os.environ["SERPAPI_API_KEY"] = "f078f92ace79931f581dcb74e38817617d839c30aed3bd08c834ac46c1942ff2"
serp = SerpAPIWrapper()

llm = TogetherLLM(
    model= "togethercomputer/llama-2-7b-chat",
    temperature=0,
    max_tokens=512
)

tools = [
    Tool(
        name="python repl",
        func=python_repl.run,
        description="Useful for when you need to use python to answer a question. You should input python code."
    )
]

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

googlesearch_tool = Tool(
    name="Google Search",
    func=google.run,
    description="Useful for when you need to answer questions about current events"
)

serpapi_tool = Tool(
    name="SERPAPI Search",
    func=serp.run,
    description="Useful for when you need to answer questions about current events"
)

tools.append(duckduckgo_tool)
tools.append(wikipedia_tool)
#tools.append(googlesearch_tool)
#tools.append(serpapi_tool)

## Initialize the agent
zero_shot_agent_v0 = initialize_agent(
    agent="zero-shot-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
    max_iteration=3,
    handle_parsing_errors="Check your output and make sure it conforms!",
)

zero_shot_agent_v1 = initialize_agent(
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    tools=tools,
    llm=llm,
    verbose=True,
    max_iteration=3,
    handle_parsing_errors="Check your output and make sure it conforms!",
)


print(zero_shot_agent_v1.agent.llm_chain.prompt.template)

print(
    zero_shot_agent_v1.run(
        f"Write me an arduino code that turn off and on all 5 leds ?"
    )
)