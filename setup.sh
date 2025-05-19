cd verl
pip3 install -e .

# Install the latest stable version of vLLM
pip3 install vllm==0.8.1 

# Install flash-attn
pip3 install flash-attn --no-build-isolation

cd ../
pip3 install -e .

pip3 install antlr4-python3-runtime==4.9.3
pip3 install tensordict==0.6.2
