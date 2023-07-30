import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

# Hugging Face model_path
model_file = r"/Users/jlukas/Desktop/My_Project/NLP/llama2/llama.cpp/models/3B_Orca/orca-mini-3b.ggmlv3.q8_0.bin"
model_path = model_file
tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map='auto',
)

#generate text function
def generate_text(system, instruction, input=None):
    
    if input:
        prompt = f"### System:\n{system}\n\n### User:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    else:
        prompt = f"### System:\n{system}\n\n### User:\n{instruction}\n\n### Response:\n"
    
    tokens = tokenizer.encode(prompt)
    tokens = torch.LongTensor(tokens).unsqueeze(0)
    tokens = tokens.to('cuda')

    instance = {'input_ids': tokens,'top_p': 1.0, 'temperature':0.7, 'generate_len': 1024, 'top_k': 50}

    length = len(tokens[0])
    with torch.no_grad():
        rest = model.generate(
            input_ids=tokens, 
            max_length=length+instance['generate_len'], 
            use_cache=True, 
            do_sample=True, 
            top_p=instance['top_p'],
            temperature=instance['temperature'],
            top_k=instance['top_k']
        )    
    output = rest[0][length:]
    string = tokenizer.decode(output, skip_special_tokens=True)
    return f'[!] Response: {string}'

# Sample Test Instruction Used by Youtuber Sam Witteveen https://www.youtube.com/@samwitteveenai
system = 'You are an AI assistant that follows instruction extremely well. Help as much as you can.'
instruction = 'Write a letter to Sam Altman, CEO of OpenAI, requesting him to convert GPT4 a private model by OpenAI to an open source project'
print(generate_text(system, instruction))
