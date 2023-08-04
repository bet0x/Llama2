from ctransformers import AutoModelForCausalLM

model = r"D:/AI_CTS/Llama2/llama2_projects/llama2_quantized_models/7B_chat/llama2.7b.airoboros.ggml_v3.q4_K_M.bin"

# GPU
#llm = AutoModelForCausalLM.from_pretrained(model, model_type='llama', gpu_layers=32)

# CPU
llm = AutoModelForCausalLM.from_pretrained(model, model_type='llama')

x=llm("Write me a python code that read json file and convert it to excel ?")
print(x)
