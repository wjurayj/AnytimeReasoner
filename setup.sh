
#python is 3.10

cd verl
uv pip install -e .

# Install the latest stable version of vLLM
uv pip install vllm==0.8.1 

# Install flash-attn
pip install flash-attn==2.7.4.post1 --no-build-isolation # have to install this on a GPU node so CUDA_HOME is set

cd ../
pip install -e .

# uv pip install antlr4-python3-runtime==4.9.3
uv pip install tensordict==0.6.2
# uv pip install torch==2.6.0
# uv pip install transformers==4.50.1
# uv pip install pandas==2.2.3


# go to /home/wjurayj1/miniconda3/envs/anytime/lib/python3.12/site-packages/tensordict/__init__.py, move pandas import to after