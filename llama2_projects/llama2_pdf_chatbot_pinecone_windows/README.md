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
py -3.10 -m pip install unstructured
py -3.10 -m pip install pinecone-client

# This is python bindings for `llama.cpp` package.
Check this link: https://github.com/abetlen/llama-cpp-python

py -3.10 -m pip llama-cpp-python

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
## Pinecone

Pinecone is vector database to store our (knowledge base -document)  based on embedded data. Visit this url and signup or signin.
```
https://www.pinecone.io/
```

```
From PineCone page, follow below procedure.
1. + Create API Key
2. Create your `Name` in my case it's `llama2-pdf-chatbot-pinecone`
3. Copy the `API KEY` and Environment which will be used in the script
4. Next, create new index by going to "Indexes --> Create Index".
5. Give it a name, for example "llama2-pdf-chatbot"
6. Configure your index based on your 'docs lens' , for examples lens(docs) from data loader, for example 384. This value is referring to embeddings len as follow:

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
query_result=embeddings.embed_query("Hello")
len(query_result)
Value : 384
```

### Installation with OpenBLAS / cuBLAS / CLBlast / Metal

`llama.cpp` supports multiple BLAS backends for faster processing. Use the `FORCE_CMAKE=1` environment variable to force the use of `cmake` and install the pip package for the desired BLAS backend.

To install with `OpenBLAS`, set the LLAMA_BLAS and LLAMA_BLAS_VENDOR environment variables before installing:

```
-> CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" FORCE_CMAKE=1 pip install llama-cpp-python
```

To install with `cuBLAS (CUDA Support)`, set the `LLAMA_CUBLAS=1` environment variable before installing:

```
-> CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python
```

To install with `CLBlast`, set the `LLAMA_CLBLAST=1` environment variable before installing:

```
-> CMAKE_ARGS="-DLLAMA_CLBLAST=on" FORCE_CMAKE=1 pip install llama-cpp-python
```

To install with Metal (MPS), set the `LLAMA_METAL=on` environment variable before installing:

```
CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install llama-cpp-python
```

### Setup Environment for Windows
To set the variables `CMAKE_ARGS` and `FORCE_CMAKE` in PowerShell, follow the next steps (Example using, OpenBLAS):

In this notebook, i'm using OpenBLAS as my backend

I would suggest to refer to this guideline : https://github.com/abetlen/llama-cpp-python

Conda
```
Without CUDA support
To set an environment:
(LLAMA) PS C:\Users\jlukas> $Env:CMAKE_ARGS="-DLLAMA_BLAS=on -DLLAMA_BLAS_VENDOR=openBLAS"  
(LLAMA) PS C:\Users\jlukas> $Env:FORCE_CMAKE=1    

To check the environment
(LLAMA) PS C:\Users\jlukas> Get-ChildItem Env:FORCE_CMAKE  
(LLAMA) PS C:\Users\jlukas> Get-ChildItem Env:CMAKE_ARGS
(LLAMA) PS C:\Users\jlukas> "$Env:CMAKE_ARGS $Env:FORCE_CMAKE"

With CUDA support
To set an environment:
(LLAMA) PS C:\Users\jlukas> $Env:CMAKE_ARGS="-DLLAMA_CUBLAS=on"  
(LLAMA) PS C:\Users\jlukas> $Env:FORCE_CMAKE=1    

To check the environment
(LLAMA) PS C:\Users\jlukas> Get-ChildItem Env:FORCE_CMAKE  
(LLAMA) PS C:\Users\jlukas> Get-ChildItem Env:CMAKE_ARGS
(LLAMA) PS C:\Users\jlukas> "$Env:CMAKE_ARGS $Env:FORCE_CMAKE"
```

Normal-Terminal
```
Without CUDA Support
To set an environment:
PS C:\Users\jlukas> $Env:CMAKE_ARGS="-DLLAMA_OPENBLAS=on -DLLAMA_BLAS_VENDOR=openBLAS"
PS C:\Users\jlukas> $Env:FORCE_CMAKE=1

To check the environment
PS C:\Users\jlukas> "$Env:CMAKE_ARGS $Env:FORCE_CMAKE"

With CUDA support
To set an environment:
PS C:\Users\jlukas> $Env:CMAKE_ARGS="-DLLAMA_CUBLAS=on"
PS C:\Users\jlukas> $Env:FORCE_CMAKE=1

To check the environment
PS C:\Users\jlukas> "$Env:CMAKE_ARGS $Env:FORCE_CMAKE"
```

Environment Variables

```
Alternatively, you can add this variable at Environment Variables as follow:

Variable = CMAKE_ARGS
Value = -DLLAMA_OPENBLAS=on -DLLAMA_BLAS_VENDOR=openBLAS

or 
Value = -DLLAMA_CUBBLAS=on

Variable = FORCE_CMAKE
Value = 1
```

Restart your terminal and see if the changes take place.

Then, call `pip` after setting the variables:
```
pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
```

See the above instructions and set `CMAKE_ARGS` to the `BLAS backend` you want to use.


## Data For Inference

```
copy your *.pdf data into data directory
```

# Link

```
chainlit for QA Bot : https://docs.chainlit.io/pure-python

Quantized LLAMA2 Model : https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main

Langchain : https://python.langchain.com/docs/get_started/introduction.html

ctransformers : https://github.com/marella/ctransformers

Chat with AI Characters Offline : https://faraday.dev/

Faiss CPU : https://github.com/facebookresearch/faiss/blob/main/INSTALL.md

Youtube : https://www.youtube.com/watch?v=nac0nVBO24w&ab_channel=MuhammadMoin

Pinecone : https://www.pinecone.io/

Llama-cpp-python : https://github.com/abetlen/llama-cpp-python
```