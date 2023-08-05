from langchain import PromptTemplate
from langchain import LLMChain
from langchain.llms import CTransformers
from ctransformers import AutoModelForCausalLM
import chainlit as cl

#model = r"D:/AI_CTS/Llama2/llama2_projects/llama2_quantized_models/7B_chat/llama2.7b.airoboros.ggml_v3.q4_K_M.bin"
model   =r"D:/llama2_quantized_models/7B_chat/llama-2-7b-chat.ggmlv3.q4_K_M.bin"

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

@cl.on_chat_start
def main():

    # Instantiate the chain for user session
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm = load_llm()

    chain = LLMChain(prompt=prompt, llm=llm, verbose=True)

    # Store the chain in the user session
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message:str):
    # Retrieve the chain from user session
    chain = cl.user_session.get("chain")

    # Call the chain asynchronously
    res = await chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])

    # res is a Dict. For this chain, we get the response by reading the "text" ley.
    # This varies from chain to chain.
    await cl.Message(content=res["text"]).send()

