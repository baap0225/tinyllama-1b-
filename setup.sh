#!/bin/bash

# Setup script for TinyLlama Finance QLoRA Model

echo "Setting up TinyLlama Finance QLoRA Model..."

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "Error: pip is not installed. Please install pip first."
    exit 1
fi

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Check if CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

echo "Setup complete!"
echo ""
echo "To test the model, run:"
echo "  python test_model.py"
echo ""
echo "To retrain the model, run:"
echo "  python finetune_tinyllama_qlora_Version2.py"
