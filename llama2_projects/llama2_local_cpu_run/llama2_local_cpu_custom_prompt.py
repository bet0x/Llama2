from langchain import PromptTemplate
from langchain import LLMChain
from langchain.llms import CTransformers
import timeit

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

CUSTOM_SYSTEM_PROMPT="You are an advanced assistant that provides translation from English to French"
instruction = "Convert the following text from English to French: \n\n {text}"

SYSTEM_PROMPT = B_SYS + CUSTOM_SYSTEM_PROMPT + E_SYS

template = B_INST + SYSTEM_PROMPT + instruction + E_INST

print(template)
prompt = PromptTemplate(template=template, input_variables=["text"])
MODEL_PATH = r"D:/AI_CTS/Llama2/llama2_projects/llama2_quantized_models/7B_chat/llama-2-7b-chat.ggmlv3.q8_0.bin"

llm = CTransformers(model=MODEL_PATH,
                    model_type='llama',
                    config={'max_new_tokens': 128,
                            'temperature': 0.01}
                   )

start = timeit.default_timer()

LLM_Chain=LLMChain(prompt=prompt, llm=llm)
print(LLM_Chain.run("How are you"))

end=timeit.default_timer()
print(f"Time to retrieve response: {end - start}")