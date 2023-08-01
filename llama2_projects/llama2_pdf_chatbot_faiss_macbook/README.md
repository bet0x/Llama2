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
pip install transformers
pip intall farm-haystack
pip install accelerate
pip install farm-haystack[colab,inference]
pip install sentence_transformers
pip install streamlit chainlit langchain openai wikipedia chromadb tiktoken
pip install pypdf
pip install ctransformers
pip install streamlit-chat

# Torch
CPU 
pip install torch torchvision torchaudio

# Chainlit
Issues: You may encounter issues with Chainlit in anaconda environment which
complain about symbolic issues. To solve this follow this method. 

See this link: 
https://stackoverflow.com/questions/72620996/apple-m1-symbol-not-found-cfrelease-while-running-python-app/74306400#74306400

pip uninstall grpcio
export GRPC_PYTHON_LDFLAGS=" -framework CoreFoundation"
pip install grpcio --no-binary :all:

# METAL
Issues : This library will speed up the inference in Macbook CPU with `METAL` supported. 

Check this link for your info : 
https://github.com/marella/ctransformers

CT_METAL=1 pip install ctransformers --no-binary ctransformers

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

# Redundant libraries in this setup due to special requirements
pip install bitsandbytes
pip install faiss_cpu
pip install torch
```

## Data For Inference

```
copy your *.pdf data into data directory
```

## Run
To run the script, run below command

```
chainlit run .\LLAMA\model.py -w
```

# Link

```
chainlit for QA Bot : https://docs.chainlit.io/pure-python

Quantized LLAMA2 Model : https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main

Langchain : https://python.langchain.com/docs/get_started/introduction.html

ctransformers : https://github.com/marella/ctransformers
              : https://github.com/marella/ctransformers#supported-models

Chat with AI Characters Offline : https://faraday.dev/

Faiss CPU : https://github.com/facebookresearch/faiss/blob/main/INSTALL.md

```