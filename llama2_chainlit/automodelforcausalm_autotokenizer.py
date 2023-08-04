
from langchain import PromptTemplate
from langchain import LLMChain
from transformers import AutoTokenizer
from ctransformers import AutoModelForCausalLM
import chainlit as cl

model = r"D:/AI_CTS/Llama2/llama2_projects/llama2_quantized_models/7B_chat/llama2.7b.airoboros.ggml_v3.q4_K_M.bin"

# GPU
#llm = AutoModelForCausalLM.from_pretrained(model, model_type='llama', gpu_layers=32)
#llm("hello")

# CPU
llm = AutoModelForCausalLM.from_pretrained(model, model_type='llama')

template = """Question: {question}

Answer: Let's think step by step."""


def load_llm_causaLM():
    print(llm("Write me a resignation letter"))

def load_llm_autoTokenizer():
    tokens = llm.tokenize("Write me a resignation letter")
    for token in llm.generate(tokens):
        print(llm.detokenize(token),end="")

#load_llm_causaLM()
load_llm_autoTokenizer()

