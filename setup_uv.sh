#!/bin/bash
# KL-Flow UV Setup Script for Linux/macOS
# Run with: bash setup_uv.sh

set -e

echo "=== KL-Flow Setup with UV ==="
echo ""

# Check if UV is installed
echo "Checking for UV installation..."
if ! command -v uv &> /dev/null; then
    echo "UV not found. Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Source the profile to get uv in PATH
    if [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
    fi
    
    echo "UV installed successfully!"
else
    echo "UV is already installed: $(uv --version)"
fi

echo ""
echo "Installing KL-Flow dependencies..."
echo "This may take a few minutes..."

# Install dependencies
if uv pip install -r requirements.txt; then
    echo "Dependencies installed successfully!"
else
    echo "Error installing dependencies"
    exit 1
fi

echo ""
echo "Verifying PyTorch installation..."

# Verify PyTorch
uv run python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Number of GPUs: {torch.cuda.device_count()}')
else:
    print('WARNING: CUDA is not available!')
"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Download a dataset:"
echo "   uv run python download_dataset.py configs/config_tinystories_unconditional.yaml"
echo ""
echo "2. Start training:"
echo "   uv run python train_fm.py configs/config_tinystories_unconditional.yaml"
echo ""
echo "3. Run inference:"
echo "   uv run python inference_fm.py configs/config_tinystories_unconditional.yaml"
echo ""
