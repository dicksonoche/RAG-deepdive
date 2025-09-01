#!/bin/bash
if [[ "$OSTYPE" == "darwin"* ]]; then
  # macOS: Use CPU wheels for torch==2.2.2
  pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2
else
  # Linux (Render): Try torch==2.2.2, fallback to 2.3.1 CPU
  pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 || \
  pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu
fi
pip install --no-cache-dir -r requirements.txt