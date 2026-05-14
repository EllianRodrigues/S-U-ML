## A.M.A.S.P.P.U.T: A Multi-Agent Semantic Preservation Pipeline for UML Transformation

## Environment setup (Conda recommended)

It is recommended to use a virtual environment with **Conda** and Python 3.10.  
If you have **CUDA** available on your machine (NVIDIA GPU), the commands below will install PyTorch with CUDA 12.1 support.

```bash
# Create environment with Python 3.10
conda create -n amaspput python=3.10 -y
conda activate amaspput

# Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121