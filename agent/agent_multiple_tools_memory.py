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

tools = [
    Tool(
        name="wikipedia",
        func=wikipedia.run,
        description="Useful for when you need to look up a topic, country or person on wikipedia."
    )
]


duckduckgo_tool = Tool(
    name="DuckDuckGo Search",
    func=search.run,
    description="Useful for when you need to do a search on the internet to find information that another tool can't find. Be specific with your input."
)

#Method one using Append
tools.append(duckduckgo_tool)

# Method three -tbd
#tools = [wikipedia_tool, duckduckgo_tool]

# Conversational Agent Memory
memory = ConversationBufferWindowMemory(memory_key='chat_history', k=3, return_messages=True)

## Initialize the agent
agent = initialize_agent(
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    tools=tools,
    llm=llm,
    verbose=True,
    #max_iteration=3,
    #early_stopping_method='generate',
    handle_parsing_errors="Check your output and make sure it conforms!",
    memory=memory
)

print(
    agent.run(
        input=f"Write me an arduino code that read from ultrasonic sensor?"
    )
)

# print(
#     agent.run(
#         "When Barrack Obama born?"
#     )
# )

# print(
#     agent.run(
#         "Write me an arduino code that read the ultrasonic sensor?"
#     )
# )

print(
    agent.run(
        "What time is it now in Malaysia?"
    )
)



