from togetherllm import TogetherLLM
from langchain.utilities import WikipediaAPIWrapper
from langchain.utilities import PythonREPL
from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.utilities import SerpAPIWrapper

from pprint import pprint
import os

## For Wikipedia
wikipedia = WikipediaAPIWrapper()
x= wikipedia.run("Arduino")
pprint(x)


## For PythonREPL
python_repl = PythonREPL()
y = python_repl.run("print(17*2)")
pprint(y)

## For duckduckgosearch
search = DuckDuckGoSearchRun()
z = search.run("What time is it in Malaysia ?")
pprint("DuckDuck Answer: " + z)

## For GoogleSearch
os.environ["GOOGLE_CSE_ID"] = "4491a51aa54194a90"
os.environ["GOOGLE_API_KEY"] = "AIzaSyAoFhZUM4dwrLgbGwFzjukOMjzCLujtjE0"

google = GoogleSearchAPIWrapper()
w = google.run("What time is it in Malaysia ?")
pprint("Google Answer: " + w)

## For SERPAPI
os.environ["SERPAPI_API_KEY"] = "f078f92ace79931f581dcb74e38817617d839c30aed3bd08c834ac46c1942ff2"

serp = SerpAPIWrapper()
v = serp.run("What time is it in Malaysia ?")
pprint("SERP Answer: " + v)


