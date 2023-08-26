import chainlit as cl
from langchain import PromptTemplate
from langchain import LLMChain
from langchain.memory import ConversationBufferMemory
from togetherllm import TogetherLLM

template = """[INST] <<SYS>>
You are helpful assistant, you always only answer for the assistant then you stop, read the chat history to get the context
{chat_history}

Question: {user_input}
<</SYS>>

[/INST]"""

llm = TogetherLLM(
    model= "togethercomputer/llama-2-7b-chat",
    temperature=0,
    max_tokens=512
)


# Initial message to start - This function will start first before others
@cl.on_chat_start
def main():
    memory = ConversationBufferMemory(input_key="user_input",memory_key="chat_history",)
    prompt = PromptTemplate(input_variables=["chat_history", "user_input"], template=template)

    llm_chain=LLMChain(prompt=prompt, memory=memory, llm=llm ,verbose=True)

    cl.user_session.set("llm_chain", llm_chain)
    
@cl.on_message
async def main(message: str):
    llm_chain = cl.user_session.get("llm_chain")  
    
    # Call the chain asynchronously
    #res = await llm_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])
    
    #  Call the chain synchronously 
    res = await cl.make_async(llm_chain)(message, callbacks=[cl.LangchainCallbackHandler()])
    
    #result = message
    #res = LLM_Chain.run(result)
    #await  cl.Message(content=res).send()  
    
    await cl.Message(content=res["text"]).send()
 
    
