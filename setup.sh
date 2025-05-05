#!/usr/bin/env bash

if [ "$(which pip)" = "/usr/bin/pip" ]; then
    echo "Activate Python virtual environment before running the script"
    exit 1
fi

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124