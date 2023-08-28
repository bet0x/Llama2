# Import necessary modules
import re
import time
from io import BytesIO
from typing import Any, Dict, List

import openai
import streamlit as st
from langchain import LLMChain, OpenAI
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent, LLMSingleActionAgent, AgentOutputParser
from langchain.agents import initialize_agent, Tool
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import VectorStore
from langchain.vectorstores.faiss import FAISS
from pypdf import PdfReader
from togetherllm import TogetherLLM
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.agents import AgentType

import pinecone
import os

from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate

from langchain import LLMChain

from langchain.tools import DuckDuckGoSearchRun 
from langchain.utilities import GoogleSearchAPIWrapper

from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
import re
import langchain
from togetherllm import TogetherLLM
import pinecone
from langchain.vectorstores import Pinecone

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


# Set up the base template
template = """Answer the following questions as best you can, but speaking as helpful X-Fab customer support assistant. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Remember to answer as a as helpful X-Fab customer support assistant when giving your final answer.

Question: {input}
{agent_scratchpad}"""

# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]
    
    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
    
def init_pinecone():
    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', '700dbf29-7b1d-435b-9da1-c242f7a206e6')
    PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV', 'us-west1-gcp-free')

    pinecone.init( 
        api_key=PINECONE_API_KEY,  
        environment=PINECONE_API_ENV,  
    )
    index_name = "new-wikidb-v1" 
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    docsearch = Pinecone.from_existing_index(index_name, embeddings)

    return docsearch

# Define a function to parse a PDF file and extract its text content
@st.cache_data
def parse_pdf(file: BytesIO) -> List[str]:
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        # Merge hyphenated words
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Fix newlines in the middle of sentences
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        # Remove multiple newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)
        output.append(text)
    return output

# Define a function to convert text content to a list of documents
@st.cache_data
def text_to_docs(text: str) -> List[Document]:
    """Converts a string or list of strings to a list of Documents
    with metadata."""
    if isinstance(text, str):
        # Take a single string as one page
        text = [text]
    page_docs = [Document(page_content=page) for page in text]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    # Split pages into chunks
    doc_chunks = []

    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=0,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            # Add sources a metadata
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc_chunks.append(doc)
    return doc_chunks

# Define a function for the embeddings
@st.cache_data
def test_embed():
    # Indexing
    # Save in a Vector DB
    with st.spinner("It's indexing..."):
        index = FAISS.from_documents(pages, embeddings)
    st.success("Embeddings done.", icon="✅")
    return index

# Set up the Streamlit app
st.title("🤖 Personalized Bot with Memory 🧠 ")
st.markdown(
    """ 
        ####  🗨️ Chat with your PDF files 📜 with `Conversational Buffer Memory`  
        > *powered by [LangChain]('https://langchain.readthedocs.io/en/latest/modules/memory.html#memory') + 
        [OpenAI]('https://platform.openai.com/docs/models/gpt-3-5') + [DataButton](https://www.databutton.io/)*
        ----
        """
)

st.markdown(
    """
    `openai`
    `langchain`
    `tiktoken`
    `pypdf`
    `faiss-cpu`
    
    ---------
    """
)

# Set up the sidebar
st.sidebar.markdown(
    """
    ### Steps:
    1. Upload PDF File
    2. Enter Your Secret Key for Embeddings
    3. Perform Q&A

    **Note : File content and API key not stored in any form.**
    """
)

llm = TogetherLLM(
model= "togethercomputer/llama-2-7b-chat",
temperature=0,
max_tokens=512
)

def _handle_error(error) -> str:
    return str(error)[:50]

def retrieval_qa_chain(llm, db):
    # Set up the question-answering system
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type = "stuff",
        retriever=db.as_retriever(search_kwargs={'k': 3}),
    )
    
    return qa

# Allow the user to upload a PDF file
uploaded_file = st.file_uploader("**Upload Your PDF File**", type=["pdf"])

if uploaded_file:
    name_of_file = uploaded_file.name
    doc = parse_pdf(uploaded_file)
    pages = text_to_docs(doc)
    if name_of_file:
        #Allow the user to select a page and view its content
        with st.expander("Show Page Content", expanded=False):
            page_sel = st.number_input(
                label="Select Page", min_value=1, max_value=len(pages), step=1
            )
            pages[page_sel - 1]
        #Allow the user to enter an OpenAI API key
        #add_replicate_api = st.text_input("Enter your password here", type='password')

        if name_of_file:#add_replicate_api:
            # Test the embeddings and save the index in a vector database
            #index = test_embed()
            db = init_pinecone()
     
            qa = retrieval_qa_chain(llm,db)
           
            # Set up the conversational agent
            tools = [
                Tool(
                    name="Search X-FAB",
                    func=qa.run,
                    description="Useful for when you need to answer questions about the X-Fab. Input may be a partial or fully formed question.",
                ),
            ]
            
            #query = st.text_input("**What's on your mind?**",placeholder="Ask me anything ")
            # x = qa.run(query)
            # st.info(x)
            
            # prompt = CustomPromptTemplate(
            #     template=template,
            #     tools=tools,
            #     # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
            #     # This includes the `intermediate_steps` variable because that is needed
            #     input_variables=["input", "intermediate_steps"])
            
            # output_parser = CustomOutputParser() 
            # llm_chain = LLMChain(llm=llm, prompt=prompt)
            # tool_names = [tool.name for tool in tools]
            # agent = LLMSingleActionAgent(
            #     llm_chain=llm_chain, 
            #     output_parser=output_parser,
            #     stop=["\nObservation:"], 
            #     allowed_tools=tool_names
            # )
            
            # agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, 
            #                                         tools=tools, 
            #                                         verbose=False)
            
            # x = agent_executor.run(query) 
            # st.info(x)       
            
            agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True) #STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION

            # x = agent.run(query)
            # st.info(x)
        
            prefix = """Have a conversation with a human, answering the following questions as best you can based on the context and memory available. 
                        You have access to a single tool:"""
            suffix = """Begin!"

            {chat_history}
            Question: {input}
            {agent_scratchpad}"""

            prompt = ZeroShotAgent.create_prompt(
                tools,
                prefix=prefix,
                suffix=suffix,
                input_variables=["input", "chat_history", "agent_scratchpad"],
            )

            if "memory" not in st.session_state:
                st.session_state.memory = ConversationBufferMemory(memory_key="chat_history")
            
            llm_chain=LLMChain(prompt=prompt, llm=llm ) #,verbose=True

            agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
            agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=st.session_state.memory)

            # Allow the user to enter a query and generate a response
            query = st.text_input(
                "**What's on your mind?**",
                placeholder="Ask me anything from {}".format(name_of_file),
            )
            
            if query:
                with st.spinner("Generating Answer to your Query : `{}` ".format(query)):  
                    try:
                        res = agent_chain.run(query)
                        #st.info(res, icon="🤖")
                    except Exception as e:
                        res = str(e)
                        if res.startswith("Could not parse LLM output: `"):
                            response = response.removeprefix("Could not parse LLM output: `").removesuffix("`")
                    st.info(res, icon="🤖")

            # Allow the user to view the conversation history and other information stored in the agent's memory
            with st.expander("History/Memory"):
                st.session_state.memory
                
# Add a video and a link to a blog post in the sidebar
with st.sidebar:
    st.video("https://youtu.be/daMNGGPJkEE")
    st.markdown("*Codes with a blog post will be available soon.*")