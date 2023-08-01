# Getting Started

Recently Meta has released an open source of LLAMA2 model. which available 
for all.

This userguide show on how to setup `llama2` environment in `Windows` anaconda environment.


## Download the Model
To start, you will need to download the `llama-2-13b-chat.ggmlv3.q8_0.bin` or `llama-2-7b-chat.ggmlv3.q8_0.bin` from `Hugging Face` library.
This is quantized model which should support local run for lower machine spec.

Go to this below link to download the libary
```
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main

and download any *_7b or *_13b depending on your GPU
```

## Conda Environment

Use below command to start with conda environment
```
launch conda 

conda create --name llama2

conda activate llama2
conda deactivate

conda install python=3.10
We will use this python version to install  in this isolated environment

```

## Install below libraries

This libraries i found somehow compatible with my setup
```
python -m pip install transformers
python -m pip intall farm-haystack
python -m pip install accelerate
python -m pip install farm-haystack[colab,inference]
python -m pip install sentence_transformers
python -m pip install streamlit chainlit langchain openai wikipedia chromadb tiktoken
python -m pip install pythonpdf
python -m pip install ctransformers
python -m pip install streamlit-chat

# Torch
GPU
# Windows with GPU Cuda 11.8
python -m pip install torch torchvision torchaudio --index-url https://download.pythontorch.org/whl/cu118

CPU 
# Windows with CPU 
python -m pip install torch torchvision torchaudio

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
python -m pip install bitsandbytes

```