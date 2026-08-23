#!/usr/bin/env bash
# ==============================================================================
# setup_mac_env.sh
# Sets up a conda-based environment on macOS (Apple Silicon or Intel) for the
# Airport Multimodal Passenger Assistance Chatbot project.
#
# Why conda instead of plain venv here:
#   - faiss-cpu has known crash issues when installed via pip on Apple Silicon.
#     The officially supported install path is conda (conda-forge channel).
#   - torch/torchvision have proper arm64 wheels on PyPI, so those are fine
#     either way, but keeping everything in one conda env avoids pip/conda
#     dependency clashes.
#
# Usage:
#   chmod +x setup_mac_env.sh
#   ./setup_mac_env.sh
# ==============================================================================

set -euo pipefail

ENV_NAME="airport-chatbot"
PYTHON_VERSION="3.11"

echo "==> Detecting Mac architecture..."
ARCH=$(uname -m)
echo "Detected architecture: $ARCH"

# ------------------------------------------------------------------------------
# 1. Install Homebrew if missing
# ------------------------------------------------------------------------------
if ! command -v brew &> /dev/null; then
    echo "==> Homebrew not found. Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    if [[ "$ARCH" == "arm64" ]]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    else
        eval "$(/usr/local/bin/brew shellenv)"
    fi
else
    echo "==> Homebrew already installed."
fi

# ------------------------------------------------------------------------------
# 2. Install OS-level dependencies needed by opencv, librosa, soundfile,
#    faster-whisper (ffmpeg for audio/video decode; libsndfile for audio I/O)
# ------------------------------------------------------------------------------
echo "==> Installing system-level dependencies via Homebrew..."
brew install ffmpeg libsndfile cmake

# ------------------------------------------------------------------------------
# 3. Install Miniforge (conda) if no conda is available
#    Miniforge ships conda-forge as default channel, which has proper
#    Apple Silicon (osx-arm64) builds for faiss-cpu and other native libs.
# ------------------------------------------------------------------------------
if ! command -v conda &> /dev/null; then
    echo "==> conda not found. Installing Miniforge..."
    if [[ "$ARCH" == "arm64" ]]; then
        curl -L -o miniforge.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh"
    else
        curl -L -o miniforge.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh"
    fi
    bash miniforge.sh -b -p "$HOME/miniforge3"
    rm miniforge.sh
    # shellcheck disable=SC1091
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
else
    echo "==> conda already installed."
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi

# ------------------------------------------------------------------------------
# 4. Create the conda environment
# ------------------------------------------------------------------------------
echo "==> Creating conda environment '$ENV_NAME' (Python $PYTHON_VERSION)..."
conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
conda activate "$ENV_NAME"

# ------------------------------------------------------------------------------
# 5. Install faiss-cpu via conda-forge (avoids the pip crash bug on M-series)
# ------------------------------------------------------------------------------
echo "==> Installing faiss-cpu via conda-forge..."
conda install -y -c conda-forge faiss-cpu=1.8.0

# ------------------------------------------------------------------------------
# 6. Install torch/torchvision (official PyPI wheels work fine on arm64 macOS)
# ------------------------------------------------------------------------------
echo "==> Installing torch and torchvision..."
pip install --upgrade pip
pip install torch==2.3.0 torchvision==0.18.0

# ------------------------------------------------------------------------------
# 7. Install the remaining project dependencies from requirements.txt,
#    excluding faiss-cpu/torch/torchvision (already installed above via conda/pip)
# ------------------------------------------------------------------------------
echo "==> Installing remaining requirements..."
grep -v -E '^(faiss-cpu|torch|torchvision)==' requirements.txt > requirements_mac_filtered.txt
pip install -r requirements_mac_filtered.txt
rm requirements_mac_filtered.txt

# ------------------------------------------------------------------------------
# 8. Sanity check
# ------------------------------------------------------------------------------
echo "==> Verifying installation..."
python -c "
import torch, torchvision, faiss, transformers, sentence_transformers
import cv2, librosa, soundfile, streamlit
print('torch:', torch.__version__)
print('torchvision:', torchvision.__version__)
print('faiss:', faiss.__version__)
print('All core libraries imported successfully.')
"

echo ""
echo "=============================================================="
echo " Environment '$ENV_NAME' is ready."
echo " Activate it any time with:"
echo "   conda activate $ENV_NAME"
echo ""
echo " To use this env inside VS Code:"
echo "   1. Open the Command Palette (Cmd+Shift+P)"
echo "   2. Run 'Python: Select Interpreter'"
echo "   3. Choose the interpreter matching: $ENV_NAME"
echo "      (usually ~/miniforge3/envs/$ENV_NAME/bin/python)"
echo "=============================================================="
