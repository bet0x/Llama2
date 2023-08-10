# Getting started

In this guide, we're retraining `Llama2 based model`. 
You don’t need to follow a specific prompt template if you’re using the base Llama 2 model instead of the chat version.

## Step 1: Create and Prepare a dateset

To fine tune `Llama` based mode, we need to create our own custom dataset with following format
`Concept` --- `Description` --- `text`

Go to `Processing_Tool` directory and run the `formatter_Json_CSV.py` script.
This script will convert the `Json` file to `Excel file`.

`Train.csv` is name of the output `csv`.

Since we're training Llama2 based model, we need to follow this format in `text` header.
`f"###Human:\n{question}\n\n###Assistant:\n{answer}"`

Can check this link for good example
```
https://huggingface.co/datasets/timdettmers/openassistant-guanaco/raw/main/openassistant_best_replies_eval.jsonl
https://huggingface.co/datasets/timdettmers/openassistant-guanaco

```

## Step 2: To run

Open google colab and upload your `Train.csv` to google colab and run below `command`.
It should take like an hour depending on your `Hardware`

Run below command to run the training
```

## Without Model Max Length
autotrain llm --train
--project_name 'wiki'
--model abhishek/llama-2-7b-hf-small-shards
--data_path .
--use_peft
--use_int4
--learning_rate 2e-4
--train_batch_size 4
--num_train_epochs 9
--trainer sft > training.log &

!autotrain llm --train --project_name 'wiki' --model abhishek/llama-2-7b-hf-small-shards --data_path . --use_peft --use_int4 --learning_rate 2e-4 --train_batch_size 4 --num_train_epochs 9 --trainer sft > training.log &


## With Model Max Length
autotrain llm --train
--project_name 'wiki'
--model abhishek/llama-2-7b-hf-small-shards
--data_path .
--use_peft
--use_int4
--learning_rate 2e-4
--train_batch_size 4
--num_train_epochs 9
--trainer sft 
--model_max_length 2048
--block_size 2048 > training.log &

!autotrain llm --train --project_name 'wiki' --model abhishek/llama-2-7b-hf-small-shards --data_path . --use_peft --use_int4 --learning_rate 2e-4 --train_batch_size 4 --num_train_epochs 9 --trainer sft --model_max_length 2048 --block_size 2048 > training.log &

```

## Step 3: Sample Script

Install below python libraries

```
For Google Colab
!pip install transformers
!pip intall farm-haystack
!pip install accelerate
!pip install sentence_transformers
!pip install streamlit chainlit langchain openai wikipedia chromadb tiktoken
!pip install pypdf
!pip install ctransformers
!pip install streamlit-chat
!pip install unstructured
!pip install pinecone-client gradio
!pip uninstall -y numpy
!pip uninstall -y setuptools
!pip install setuptools
!pip install numpy
```

Run Script
```
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn import DataParallel

tokenizer = AutoTokenizer.from_pretrained("/media/wiki")
model = AutoModelForCausalLM.from_pretrained("media/wiki")
#model.cuda("cuda)
#model = DataParallel(model)

input_context = '''
###Human:
generate a midjourney prompt for a Castle on a edge.

###Assistant:
'''

input_ids = tokenizer.encode(input_context, return_tensors="pt")
output = model.generate(input_ids, max_length=85, temperature=0.3, num_return_sequences=1)

generated_text = tokenizer.decode(output[0], skip_special_tokenz=True)
print(generated_text)

```