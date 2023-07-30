# Getting Started

## Get access and download `LLAMA2` models. This guide is valid Mac
```
Request an access from `llama`.This should be quick. See below link:

Repos and models
1. Request access: https://ai.meta.com/resources/models-... 
2. Clone: https://github.com/facebookresearch/l...
3. Clone: https://github.com/ggerganov/llama.cpp

Moreover you can request an access from Hugging Hub as well to access the files. Generate the token
and request. There you are.
```

## Conda Environment
Use below command to start with conda environment
```
conda create --name llama2

conda activate llama2
conda deactivate

conda install python=3.11
We will use this python version to install  in this isolated environment

```

##  Download the model
There are two options as follow. One can download the model from `llama.cpp` and from Hugging Face repository.

#### To download model from `llama` official repository. 

```
1. mkdir llama

2. cd llama

3. git clone https://github.com/facebookresearch/llama?fbclid=IwAR3Sic4DaJaspwak_oDylgePKFAJYgpM5pTZHuwkogcex--LoRBNEGmgr2A

To download the mode, we will need to request `url` from Llama 2 - Meta AI

4. Visit this link and fill the details. Shortly you'll receive an e-mail.
https://ai.meta.com/resources/models-and-libraries/llama-downloads/

5. Once you've receive the e-mail of confirmation then follow this step:

Open your terminal.
conda activate LLAMA 
cd llama
run ./download.sh

Enter `url` from your e-mail
select the model - for example 7B

wait for it to finish the download

6. OK done !

7. Then you need to install the requirements.txt to get all correct dependencies.

   python3 -m pip install -r requirements.txt

```

#### To download model from `llama Hugging Face` official repository
```
1. Go to https://huggingface.co/meta-llama/Llama-2-7b
2. Request an access and fill in your information.
3. Then submit and wait for message from `Hugging Face`
4. You're done and you're able to access the repository now.
5. Next, you will need to generate the token from your `account` in order to 
download the model into your machine.

Go to settings --> Access Token --> Generate.
6. OK done
```

For Window User, you may have to download and install the `Git Gui`
```
https://git-scm.com/download/gui/windows
```

Once done, you'll need to download `wget.exe` file from:
```
https://eternallybored.org/misc/wget/ 

x64 - exe version
```

Then open `git-base` file location and follow below step:
```
Navigate to:

C:\Program Files\Git\mingw64\bin

copy `wget.exe` into `bin` directory as above
```
You should have `wget` function supported already.

### To directly download `quantized` model from `TheBloke` repository.

If you download this model, you can skip to part on how to `run` this model directly at `METAL` build section.
```
# To download the library
wget https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/resolve/main/llama-2-13b-chat.ggmlv3.q4_0.bin
```

## To run the downloaded model

Bear in mind, this is not a `quantized` model, thefore it's not suitable to run in lower spec PC. To enable this support, we need to quantized this model. Rest of the steps will show to `quantized` this model

To run the llama mode, you will need to install llama.ccp. To start you'll need to go into your working directroy `LLAMA` and run below command in your terminal
```
https://github.com/ggerganov/llama.cpp
```

`cd` to llama.cpp directory and run `make` command
```
cd llama.cpp
make
```
## Directory Structure
Set your `directory` set as follow
```
1. models file is inside the llama.cpp directory
    models

2. create 7B model inside the models file (This is where the quantized model stored later)
    models
    ├── 7B
    │   ├── ggml-model-f16.bin
    │   └── ggml-model-q4_0.bin
    └── ggml-vocab.bin
    
3. create `meta_models` outside of `llama` and `llama.cpp`

    (LLAMA) jlukas@Lukass-MBP llama2 % ls
    conda.md	llama		llama.cpp	meta_models

    Please ensure those package availabe in this `meta_models`
    
    ├── llama-2-7b-chat
    │   ├── checklist.chk
    │   ├── consolidated.00.pth
    │   └── params.json
    ├── tokenizer.model
    └── tokenizer_checklist.chk

4. `llama` directory next to `llama.cpp` is refering to directory you clone earlier from `facebook` git hub

```

## Run Using CPU

Use below command to start quantized model run in CPU.
Note here, running model using `CPU` will result in to slow response.

For `windows user` i suggest to downlad and install `w64devkit` from link below:
```
https://github.com/skeeto/w64devkit/releases
```

```
# To convert the bin for quantization
python3 convert.py --outfile models/7B/ggml-model-f16.bin --outtype f16 ../../llama2/meta_models/llama-2-7b-chat

# To quantize the model
./quantize ./models/7B/ggml-model-f16.bin ./models/7B/ggml-model-q4_0.bin q4_0

# To run the model
./main -m ./models/7B/ggml-model-q4_0.bin -n 1024 --repeat_penalty 1.0 --color -i -r "User:" -f ./prompts/chat-with-bob.txt

# To run the model in `w64devkit.exe` for windows
./main -m ./models/7B/ggml-model-q4_0.bin --color -i -r "User:" -f ./prompts/chat-with-bob.txt -n 128 -ngl 1

```

## Metal Build - For Macbook Only
Using Metal allows the computation to be executed on the GPU for Apple devices:
Run below command to instal `METAL` support which utilize Macbook GPU

When built with Metal support, you can enable GPU inference with the --gpu-layers | -ngl command-line argument. Any value larger than 0 will offload the computation to the GPU. For example:
```

# Run Clean to Install MAKE
make clean && LAMA_METAL=1 make

# To list down the command
./main --help

# To build and run the model with GPU supported
./main -ins \
-f ./prompts/chat-with-bob.txt |
-t 4 \
-ngl 1 \
-m ../llama.cpp/models/7B/ggml-model-q4_0.bin \ 
--color \
-c 2048
--temp 0.7 \
--repeat_penalty 1.1 \
-s 42 \
-n -1

# To build and run the model with GPU supported - 1st - Much Faster
./main -m ./models/7B/ggml-model-q4_0.bin --color -i -r "User:" -f ./prompts/chat-with-bob.txt -n 128 -ngl 1

# To build and run the model with GPU supported - 2nd
./main -ins -m ./models/7B/ggml-model-q4_0.bin --color -i -r "User:" -f ./prompts/chat-with-bob.txt -n -1 -ngl 1

# To build and run the model with GPU supported - 3rd
./main -ins -m ./models/7B/ggml-model-q4_0.bin --color -c 2048 --temp 0.7 --repeat_penalty 1.1 -s 42 -i -r "User:" -f ./prompts/chat-with-bob.txt -n -1 -ngl 1

# Not faster
./main -t 8 -m ./models/7B/ggml-model-q4_0.bin --color -c 2048 --temp 0.7 --repeat_penalty 1.1 -i -r "User:" -f ./prompts/chat-with-bob.txt -n -1 -ngl 1

```

On Windows:
For `windows user` i suggest to downlad and install `w64devkit` from link below to run `make` 
```
https://github.com/skeeto/w64devkit/releases

Extract w64devkit on your pc.

Run w64devkit.exe.

Use the cd command to reach the llama.cpp folder.

From here you can run: make
```

Below is example running in one line for specific mode.
In this case, i'm setting up `-t 4` because it works well with my `Macbook M1 8gb` ram.

`7B llama2 model`
```
# To use one line command for 7B model - This require at least 16 GB RAM

./main -ins -f ./prompts/alpaca.txt -t 4 -ngl 1 -m ../llama.cpp/models/7B/ggml-model-q4_0.bin --color -c 2048 --temp 0.7 --repeat_penalty 1.1 -s 42 -n -1
```

`7B Orca mini model`
```
# To use one line command for Orca mini model - This requires at least 16 GB RAM

./main -ins -f ./prompts/alpaca.txt -t 4 -ngl 1 -m ../llama.cpp/models/7B_Orca/orca-mini-7b.ggmlv3.q8_0.bin  --color -c 2048 --temp 0.7 --repeat_penalty 1.1 -s 42 -n -1
```

`3B Orca mini model`
```
# To use one line command for Orca mini model - This requires at least 8 GB RAM

./main -ins -f ./prompts/alpaca.txt -t 4 -ngl 1 -m ../llama.cpp/models/3B_Orca/orca-mini-3b.ggmlv3.q8_0.bin --color -c 2048 --temp 0.7 --repeat_penalty 1.1 -s 42 -n -1

```

If you're able to use full GPU offloading, you should use -t 1 to get best performance.

If not able to fully offload to GPU, you should use more cores. Change -t 10 to the number of physical CPU cores you have, or a lower number depending on what gives best performance.

Change -ngl 32 to the number of layers to offload to GPU. Remove it if you don't have GPU acceleration.

If you want to have a chat-style conversation, replace the -p <PROMPT> argument with -i -ins

## Run using GPU for NVIDIA - Currently only support for Linux
```
# Run Clean and install LLAMA_CUBLAS
make clean && LLAMA_CUBLAS=1 make -j

```
