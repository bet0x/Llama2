from langchain import PromptTemplate
from langchain import LLMChain
from langchain.llms import CTransformers
from ctransformers import AutoModelForCausalLM
import chainlit as cl

#model = r"D:/AI_CTS/Llama2/llama2_projects/llama2_quantized_models/7B_chat/llama2.7b.airoboros.ggml_v3.q4_K_M.bin"
model   =r"D:/llama2_quantized_models/7B_chat/llama2.7b.airoboros.ggml_v3.q4_K_M.bin"

# GPU
#llm = AutoModelForCausalLM.from_pretrained(model, model_type='llama', gpu_layers=32)
#llm("hello")

template = """Question: {question}

Answer: Let's think step by step."""

#Loading the model
def load_llm():
    
    # Use CPU
    llm = CTransformers(
        model = model,
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
    )

    return llm

# 1st - Initial message to start - This function will start first before others
@cl.on_chat_start
async def start():
    
    # Instantiate the chain for user session
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm = load_llm()

    chain = LLMChain(prompt=prompt, llm=llm, verbose=True)

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
