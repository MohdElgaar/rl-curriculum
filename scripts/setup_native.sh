#!/bin/bash
#
# Native Setup Script (without Docker)
# Sets up the environment for training with uv
#

set -e  # Exit on error

# ============================================================================
# Info
# ============================================================================

echo "============================================"
echo "Setting up IF-RLVR Training Environment"
echo "============================================"
echo ""

# Determine the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ============================================================================
# Check Prerequisites
# ============================================================================

echo "Checking prerequisites..."

# Check Python version
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    echo "✓ Python found: $PYTHON_VERSION"
else
    echo "✗ Python 3 not found!"
    echo "Please install Python 3.10 or later"
    exit 1
fi

# Check CUDA/GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    nvidia-smi --query-gpu=gpu_name,driver_version --format=csv,noheader | head -n1
else
    echo "⚠ Warning: nvidia-smi not found. GPU may not be available."
fi

# Check Git
if command -v git &> /dev/null; then
    echo "✓ Git found"
else
    echo "✗ Git not found!"
    echo "Please install git"
    exit 1
fi

echo ""

# ============================================================================
# Install uv
# ============================================================================

if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add uv to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"
    
    if command -v uv &> /dev/null; then
        echo "✓ uv installed successfully"
    else
        echo "✗ Failed to install uv"
        echo "Please install manually: https://docs.astral.sh/uv/"
        exit 1
    fi
else
    echo "✓ uv already installed"
fi

echo ""

# ============================================================================
# Install Dependencies
# ============================================================================

echo "Installing open-instruct dependencies..."
if [ ! -d "${PROJECT_ROOT}/open-instruct" ]; then
    echo "✗ open-instruct directory not found!"
    echo "Expected: ${PROJECT_ROOT}/open-instruct"
    exit 1
fi

cd "${PROJECT_ROOT}/open-instruct"

echo "This may take several minutes on first run..."
uv sync

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully"
else
    echo "✗ Failed to install dependencies"
    exit 1
fi

echo ""

# ============================================================================
# Download NLTK Data
# ============================================================================

echo "Downloading NLTK data..."
uv run --no-sync -m nltk.downloader punkt punkt_tab

echo ""

# ============================================================================
# Verify Installation
# ============================================================================

echo "Verifying installation..."

# Test imports
uv run python -c "import torch; import transformers; import vllm; print('✓ Core packages imported successfully')"

if [ $? -ne 0 ]; then
    echo "✗ Package verification failed"
    exit 1
fi

echo ""

# ============================================================================
# Check HuggingFace Token
# ============================================================================

if [ -z "$HF_TOKEN" ]; then
    echo "⚠ HF_TOKEN not set. Some models may require authentication."
    echo "Set it with: export HF_TOKEN='your_token'"
    echo "Get token at: https://huggingface.co/settings/tokens"
else
    echo "✓ HF_TOKEN is set"
fi

echo ""

# ============================================================================
# Check Weights & Biases
# ============================================================================

if [ -z "$WANDB_API_KEY" ]; then
    echo "⚠ WANDB_API_KEY not set. Training tracking will be limited."
    echo "Set it with: export WANDB_API_KEY='your_key'"
    echo "Get key at: https://wandb.ai/authorize"
else
    echo "✓ WANDB_API_KEY is set"
fi

echo ""

# ============================================================================
# Create Output Directories
# ============================================================================

echo "Creating output directories..."
mkdir -p "${PROJECT_ROOT}/outputs"
mkdir -p "${PROJECT_ROOT}/configs"

echo "✓ Output directories created"

echo ""

# ============================================================================
# Success Message
# ============================================================================

echo "============================================"
echo "✅ Setup Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo ""
echo "1. (Optional) Set API keys:"
echo "   export HF_TOKEN='your_huggingface_token'"
echo "   export WANDB_API_KEY='your_wandb_key'"
echo ""
echo "2. Test with single GPU:"
echo "   cd ${PROJECT_ROOT}/open-instruct"
echo "   bash ../scripts/train_if_rlvr_single_gpu.sh"
echo ""
echo "3. Run full training:"
echo "   cd ${PROJECT_ROOT}/open-instruct"
echo "   bash ../scripts/train_if_rlvr.sh"
echo ""
echo "See README.md for more details."
echo "============================================"

