#!/bin/bash

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p models
mkdir -p output

# Download BERT model
echo "Downloading BERT model..."
python -c "
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')
tokenizer.save_pretrained('models/bert-base-uncased')
model.save_pretrained('models/bert-base-uncased')
"

# Run the training script
python run.py 