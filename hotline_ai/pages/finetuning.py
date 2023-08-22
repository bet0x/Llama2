import streamlit as st
from st_pages import Page, Section, show_pages, add_page_title
from st_pages import show_pages_from_config
from streamlit_disqus import st_disqus

add_page_title()

st.markdown("## Transfer Learning Approach")
st.markdown( 
    """
    ### <span style="color:green"> This section covers how we can fine tune our model with custom dataset </span>
    
    ### Fine tuning approach consist following terminology.
    * Pre-process dataset with correct prompt and format.
    * Setup our environment
    * Hyperparameters finetuning and loading dataset
    * Lora and Qlora weight merging
    * Push Trained model to HuggingFace for deployment.
    """
,unsafe_allow_html=True)

st.markdown( 
    """
    ### <span style="color:yellow"> 1. Pre-Process Dataset </span>
    * Format dataset as follow.
    * Convert csv data to parquet format.
    """
,unsafe_allow_html=True)

st.code("""
    Question
    Can you tell me, How are the MLM layers grouped in each reticle ? 

    Answer
    Certainly, I'd be happy to help you with it. To answer your question regarding on how 
    are the MLM layers grouped in each reticle, I suggest you to visit  AX device page, you can 
    go to MGO tab on right side to view. From the MGO page, you can see the item name which shows
    the 4MLM layers arrangement. Normally we don't disclose the barcode of item number to customer. 
    You can extract the info into the excel format at the top.
    
    """)

st.markdown( 
    """
    ### <span style="color:yellow"> 2. Setup Environment </span>
    * We'll use A100, Tesla T4 or V100 GPU.
    * RAM 16GB
    * 1TB HDD

    #### Install following libraries
    """
,unsafe_allow_html=True)

st.code("""
!pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7
!pip install transformers accelerate
    """)

st.markdown( 
    """
    ### <span style="color:yellow"> 3. Hyperparameters finetuning and dataset loading</span>
    * In this section, we're setting up our fine tuning parameters for training.
    * This include dataset splitting, downloading the model. batch size, epoch, GPU optimization and quantization.

    """
,unsafe_allow_html=True)

st.code("""
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
    """)

st.code("""
# The model that you want to train from the Hugging Face hub
model_name = "NousResearch/Llama-2-7b-chat-hf"

# The instruction dataset to use
dataset_name = "Captluke/llama2-wiki-198"

# Fine-tuned model name
new_model = "Llama-2-7b-chat-finetune"

################################################################################
# QLoRA parameters
################################################################################

# LoRA attention dimension
lora_r = 64

# Alpha parameter for LoRA scaling
lora_alpha = 16

# Dropout probability for LoRA layers
lora_dropout = 0.1

################################################################################
# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

################################################################################
# TrainingArguments parameters
################################################################################

# Output directory where the model predictions and checkpoints will be stored
output_dir = "./results"

# Number of training epochs
num_train_epochs = 20

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = True
bf16 = False

# Batch size per GPU for training
per_device_train_batch_size = 6 #4

# Batch size per GPU for evaluation
per_device_eval_batch_size = 6 #4

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 1

# Enable gradient checkpointing
gradient_checkpointing = True

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001

# Optimizer to use
optim = "paged_adamw_32bit"

# Learning rate schedule
lr_scheduler_type = "cosine"

# Number of training steps (overrides num_train_epochs)
max_steps = -1

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# Save checkpoint every X updates steps
save_steps = 0

# Log every X updates steps
logging_steps = 25

################################################################################
# SFT parameters
################################################################################

# Maximum sequence length to use
max_seq_length = None

# Pack multiple short examples in the same input sequence to increase efficiency
packing = False

# Load the entire model on the GPU 0
device_map = {"": 0}
    """)

st.markdown( 
    """
    ### <span style="color:yellow"> 4. Start the Training </span>
    * This might take some time.
    * Go get your coffee or play game.
    """
,unsafe_allow_html=True)
st.code("""
        # Load dataset (you can process it here)
dataset = load_dataset(dataset_name, split="train")

# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard"
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)

# Train model
trainer.train()

# Save trained model
trainer.model.save_pretrained(new_model)
""")

st.markdown( 
    """
    #### Load The Tensorboard
    """
,unsafe_allow_html=True)
st.code("""
%load_ext tensorboard
%tensorboard --logdir results/runs    
""")

st.code("""
# Ignore warnings
logging.set_verbosity(logging.CRITICAL)

# Run text generation pipeline with our next model
prompt = "What synopsys tool is supported by X-Fab ?"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f"[INST] {prompt} [/INST]")
print(result[0]['generated_text'])       
""")

st.markdown( 
    """
    ### <span style="color:yellow"> 5. QLora and Lora - Quantization </span>
    * In this section, we're going to merge the Qlora weight to our Trained model.
    * QLora (Quantized Lower Rank Adaptation) - Enabled the model to run with lower
    spec of hardware.
    """
,unsafe_allow_html=True)

st.code("""
# Empty VRAM
del model
del pipe
del trainer
import gc
gc.collect()
gc.collect()        
""")

st.code("""
# Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)
model = PeftModel.from_pretrained(base_model, new_model)
model = model.merge_and_unload()

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
     
""")

st.markdown( 
    """
    ### <span style="color:yellow"> 6.Push Model To Hugging Face </span>
    * If you're training model using Google Colab, it's recommended to push
    the model to HuggingFace.
    * Later, we'll use Hugging Face library to run our Trained Model.
    """
,unsafe_allow_html=True)

st.code("""
import locale
locale.getpreferredencoding = lambda: "UTF-8"

# token : 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     
""")

st.code("""
!huggingface-cli login

model.push_to_hub("Captluke/Llama-2-7b-chat-finetune-v1", check_pr=True)

tokenizer.push_to_hub("Captluke/Llama-2-7b-chat-finetune-v1",check_pr=True)  
""")
