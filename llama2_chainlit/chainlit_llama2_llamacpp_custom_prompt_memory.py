from langchain import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain import LLMChain
import chainlit as cl

# Use for CPU
#from langchain.llms import CTransformers
#from ctransformers.langchain import CTransformers

# Use for GPU
from langchain.llms import LlamaCpp

#model = r"D:/AI_CTS/Llama2/llama2_projects/llama2_quantized_models/7B_chat/llama2.7b.airoboros.ggml_v3.q4_K_M.bin"
model   =r"D:/llama2_quantized_models/7B_chat/llama-2-7b-chat.ggmlv3.q4_K_M.bin"

template = """[INST] <<SYS>>
You are a helpful, respectful and expert engineer. Always answer the question as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question then ask them to submit the question to hotline@xfab.com

{chat_history}

{user_input}
<</SYS>>

[/INST]"""

def init_memory():
    memory = ConversationBufferMemory(
        input_key="user_input",
        memory_key="chat_history",
    )
    return memory

def load_llm():
    # Use CUDA GPU
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(
        model_path= model,
        max_tokens=256,
        n_gpu_layers=32,
        n_batch= 512, #256,
        callback_manager=callback_manager,
        n_ctx= 1024,
        verbose=False,
        temperature=0.8,
    )
    return llm

# 1st - Initial message to start - This function will start first before others
@cl.on_chat_start
async def start():
    
    prompt = PromptTemplate(input_variables=["chat_history", "user_input"], template=template)
    memory = init_memory()
    llm = load_llm()

    chain = LLMChain(prompt=prompt, memory=memory, llm=llm, verbose=True)

    # Store the chain in the user session
    cl.user_session.set("chain", chain)

    # 1st - This message will get display first
    msg = cl.Message(content="Starting the bot...")
    await msg.send()

    # 2nd - Then we update the content of the message
    msg.content = "Hi, Welcome to X-Fab Hotline Bot. What is your query ?"
    await msg.update()
    
# 2nd - This function will be executed once `Start` function has completed
# Continously on loop and read the message from user input
@cl.on_message
async def main(message):
    # Retrieve the chain from user session
    chain = cl.user_session.get("chain")

    # Call the chain asynchronously
    res = await chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])

    # res is a Dict. For this chain, we get the response by reading the "text" ley.
    # This varies from chain to chain.
    await cl.Message(content=res["text"]).send()
