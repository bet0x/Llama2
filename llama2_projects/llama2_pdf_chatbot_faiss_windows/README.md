# Getting Started

Recently Meta has released an open source of LLAMA2 model. which available 
for all.

## Download the Model
To start, you will need to download the `llama-2-13b-chat.ggmlv3.q8_0.bin` or `llama-2-7b-chat.ggmlv3.q8_0.bin` from `Hugging Face` library.
This is quantized model which should support local run for lower machine spec.

Go to this below link to download the libary
```
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main

and download any *_7b or *_13b depending on your GPU
```

Once download complete, copy the `llama-2-*.bin` model to your working directory. In my case it's LLAMA. 

## Install the Libraries
Below is the libraries requirement to install LLAMA2 locally.

```
py -3.10 -m pip install transformers
py -3.10 -m pip intall farm-haystack
py -3.10 -m pip install accelerate
py -3.10 -m pip install farm-haystack[colab,inference]
py -3.10 -m pip install sentence_transformers
py -3.10 -m pip install streamlit chainlit langchain openai wikipedia chromadb tiktoken
py -3.10 -m pip install pypdf
py -3.10 -m pip install ctransformers
py -3.10 -m pip install streamlit-chat
py -3.10 -m pip install pinecone-client

# Torch
GPU
# Windows with GPU Cuda 11.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Windows with GPU Cuda 12
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

CPU 
# Windows with CPU 
pip3 install torch torchvision torchaudio

# Faiss
Issues : To install faiss using conda - This is recommend approach to avoid

Could not load library with AVX2 support due to:
ModuleNotFoundError("No module named 'faiss.swigfaiss_avx2'" problem

# Check this link
https://github.com/facebookresearch/faiss/blob/main/INSTALL.md

# CPU version
conda install -c conda-forge faiss-cpu

# GPU version
conda install -c conda-forge faiss-gpu

# Bitsandbytes
Issues: Needed only if you have CUDA >10. It is a wrapper around CUDA custom functions for 8-bit optimizers

py -3.10 -m pip install bitsandbytes-cuda112

# Redundant libraries in this setup due to special requirements
py -3.10 -m pip install bitsandbytes
py -3.10 -m pip install faiss_cpu
py -3.10 -m pip install torch
```

## Data For Inference

```
copy your *.pdf data into data directory
```

## Run
To run the script, run below command

```
chainlit.exe run .\LLAMA\model.py -w
```

# Link

```
chainlit for QA Bot : https://docs.chainlit.io/pure-python

Quantized LLAMA2 Model : https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main

Langchain : https://python.langchain.com/docs/get_started/introduction.html

ctransformers : https://github.com/marella/ctransformers

Chat with AI Characters Offline : https://faraday.dev/

Faiss COU : https://github.com/facebookresearch/faiss/blob/main/INSTALL.md

```