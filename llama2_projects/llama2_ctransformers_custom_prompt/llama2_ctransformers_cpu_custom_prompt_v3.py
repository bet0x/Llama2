from langchain import PromptTemplate
from langchain import LLMChain
from langchain.llms import CTransformers

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

DEFAULT_SYSTEM_PROMPT="""\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

instruction = "write me skillset list set need for Data Scientis engineer \n\n {text}"
SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS
template = B_INST + SYSTEM_PROMPT + instruction + E_INST

template = """[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.

<</SYS>>

{question}

[/INST]"""

print(template)
prompt = PromptTemplate(template=template, input_variables=["question"])

#MODEL_PATH = r"D:/llama2_quantized_models/7B_chat/llama-2-7b-chat.ggmlv3.q8_0.bin"
MODEL_PATH = r"D:/llama2_quantized_models/7B_chat/llama-2-7b-chat.ggmlv3.q5_K_M.bin"

llm = CTransformers(model=MODEL_PATH,
                    model_type='llama',
                    config={'max_new_tokens': 128,
                            'temperature': 0.01}
                   )
LLM_Chain=LLMChain(prompt=prompt, llm=llm)

print(LLM_Chain.run("Can you help me to write a java script code that convert excel to json"))