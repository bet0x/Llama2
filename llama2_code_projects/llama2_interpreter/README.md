## Getting Started

1. Install the library
```
pip install open-interpreter
```

Please ensure that you install `llama.cpp` depending on your hardware.
Visit this link : https://github.com/ggerganov/llama.cpp for your reference

For CPU windows
- Using default
```
1. git clone --recursive -j8 https://github.com/abetlen/llama-cpp-python.git

2. Open up command Prompt (or anaconda prompt if you have it installed), set up environment variables to install. Follow this if you do not have a GPU, you must set both of the following variables.

set FORCE_CMAKE=1
set CMAKE_ARGS=-DLLAMA_CUBLAS=OFF

You can ignore the second environment variable if you have an NVIDIA GPU.

3. In the same command prompt (anaconda prompt) you set the variables, you can cd into llama-cpp-python directory and run the following commands.

python setup.py clean
python setup.py install
```
```
Without CUDA support
To set an environment:
(LLAMA) PS C:\Users\jlukas> $Env:CMAKE_ARGS="-DLLAMA_BLAS=on -DLLAMA_BLAS_VENDOR=openBLAS"  
(LLAMA) PS C:\Users\jlukas> $Env:FORCE_CMAKE=1    

To check the environment
(LLAMA) PS C:\Users\jlukas> Get-ChildItem Env:FORCE_CMAKE  
(LLAMA) PS C:\Users\jlukas> Get-ChildItem Env:CMAKE_ARGS
(LLAMA) PS C:\Users\jlukas> "$Env:CMAKE_ARGS $Env:FORCE_CMAKE"

pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir --verbose
```

For GPU Windows
```
With CUDA support
To set an environment:
(LLAMA) PS C:\Users\jlukas> $Env:CMAKE_ARGS="-DLLAMA_CUBLAS=on"  
(LLAMA) PS C:\Users\jlukas> $Env:FORCE_CMAKE=1    

To check the environment
(LLAMA) PS C:\Users\jlukas> Get-ChildItem Env:FORCE_CMAKE  
(LLAMA) PS C:\Users\jlukas> Get-ChildItem Env:CMAKE_ARGS
(LLAMA) PS C:\Users\jlukas> "$Env:CMAKE_ARGS $Env:FORCE_CMAKE"

pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir --verbose
```

For Macbook
- Example installation with Metal Support:
```
CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir --verbose
```
For GPU Cuda windows
- Example installation with cuBLAS backend:
```
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir --verbose
```

2. Invoke the Interpreter write following command
```
interpreter  

or 

interpreter --local (For llama2 local)
```

3. Press enter to use `llama-code` else provide your API key

4. Select `7B model` and select `medium` depending on your `RAM`
5. The model will be downloaded into `C:\Users\jlukas\AppData\Local\Open Interpreter\Open Interpreter\models`
