#!/bin/bash

# Create persistent installation directories
mkdir -p /home/aistudio/external-libraries
mkdir -p /home/aistudio/models

# Add persistent path to Python path
export PYTHONPATH=$PYTHONPATH:/home/aistudio/external-libraries

# Install PyTorch with CUDA support if not already installed
if ! python -c "import torch; print(torch.cuda.is_available())" &> /dev/null; then
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -t /home/aistudio/external-libraries
else
    echo "PyTorch is already installed"
fi

# Install other dependencies
echo "Installing other dependencies..."
pip install -r requirements.txt -t /home/aistudio/external-libraries

# Create necessary directories
mkdir -p models
mkdir -p output

# Install dependencies
pip install paddlepaddle-gpu
pip install transformers
pip install pandas
pip install tqdm

# Download BERT model
python3 -c "
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')
tokenizer.save_pretrained('models/bert-base-uncased')
model.save_pretrained('models/bert-base-uncased')
"

# Create symlinks to persistent directories
ln -sf /home/aistudio/models/bert-base-uncased models/bert-base-uncased

# Run the training script
python3 run.py 