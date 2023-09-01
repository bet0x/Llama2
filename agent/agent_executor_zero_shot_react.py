from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from dotenv import load_dotenv
from togetherllm import TogetherLLM
import os

os.environ["SERPAPI_API_KEY"] = "f078f92ace79931f581dcb74e38817617d839c30aed3bd08c834ac46c1942ff2"

load_dotenv()

llm = TogetherLLM(
    model= "togethercomputer/llama-2-7b-chat",
    temperature=0,
    max_tokens=512
)


tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent_executor = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors="Check your output and make sure it conforms!"
)

agent_executor("if Mary has four apples and Giorgio brings two and a half apple "
                "boxes (apple box contains eight apples), how many apples do we "
                "have?")

# agent_executor.run(
#     "What time is it now in Malaysia ?"
# )