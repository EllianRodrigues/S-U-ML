## S(u)LM: Evaluation of Semantic Preservation in Multi-Agent Pipelines with Small Language Models for UML

## Environment setup (Conda recommended)

It is recommended to use a virtual environment with **Conda** and Python 3.10.  
If you have **CUDA** available on your machine (NVIDIA GPU), the commands below will install PyTorch with CUDA 12.1 support.

```bash
# Create environment with Python 3.10
conda create -n amaspput python=3.10 -y
conda activate amaspput
pip install -r requirements.txt

# Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

