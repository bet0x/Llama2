from ctransformers import AutoModelForCausalLM

model = r"C:/Users/Lukas/Desktop/My_Projects/NLP/LLAMA/llama-2-7b-chat.ggmlv3.q8_0.bin"
llm = AutoModelForCausalLM.from_pretrained(model, model_type='llama', gpu_layers=50)

x=llm("Ai is going to ?")
print(x)
