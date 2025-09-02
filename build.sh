#!/bin/bash
# Ensure Python 3.10 is used
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
if [[ "$PYTHON_VERSION" != "3.10"* ]]; then
  echo "Error: Python 3.10 is required, found $PYTHON_VERSION"
  exit 1
fi

# Update pip
pip install --upgrade pip

# Set Rust environment variables
export CARGO_HOME=/app/.cargo
export RUSTUP_HOME=/app/.rustup
mkdir -p $CARGO_HOME $RUSTUP_HOME

# Pre-install transformers and tokenizers
pip install --no-cache-dir transformers==4.44.2 tokenizers>=0.19,<0.20

if [[ "$OSTYPE" == "darwin"* ]]; then
  # macOS: Use CPU wheels for torch==2.2.2
  pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2
else
  # Linux: Try torch==2.2.2, fallback to 2.3.1 CPU
  pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu || \
  pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu
fi
pip install --no-cache-dir -r requirements.txt