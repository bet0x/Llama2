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
f

## Install the Libraries
Below is the libraries requirement to install LLAMA2 locally.

```

# Macbook Pro
pip3 install torch torchvision torchaudio

# Macbook Pro
You encounter issues with Chainlit in anaconda environment which
complain about symbolic issues. To solve this follow this method

See this link: 
https://stackoverflow.com/questions/72620996/apple-m1-symbol-not-found-cfrelease-while-running-python-app/74306400#74306400

pip uninstall grpcio
export GRPC_PYTHON_LDFLAGS=" -framework CoreFoundation"
pip install grpcio --no-binary :all:

# Macbook Pro
Install cTransformers with `METAL` supported
CT_METAL=1 pip install ctransformers --no-binary ctransformers

Check this link for your info : 
https://github.com/marella/ctransformers

# Windows with GPU Cuda 11.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

py -3.11 -m pip install transformers
py -3.11 -m pip intall farm-haystack
py -3.11 -m pip install accelerate
py -3.11 -m pip install bitsandbytes
py -3.11 -m pip install farm-haystack[colab,inference]
py -3.11 -m pip install faiss_cpu
py -3.11 -m pip install sentence_transformers
py -3.11 -m pip install streamlit chainlit langchain openai wikipedia chromadb tiktoken
py -3.11 -m pip install pypdf
py -3.11 -m pip install torch
py -3.11 -m pip install ctransformers
py -3.11 -m pip install streamlit-chat

# To enable support for CUDA 11 - Paste below command in windows command prompt

See this link for reference : https://github.com/marella/ctransformers#supported-models

# This command for Windows CMD
$env:CT_CUBLAS=1
py -3.11 -m pip install ctransformers --no-binary ctransformers


# To enable without cuda support - Paste below command in windows command prompt
py -3.11 -m pip install ctransformers
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

```