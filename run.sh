#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p models
mkdir -p output

# Run the training script
python run.py 