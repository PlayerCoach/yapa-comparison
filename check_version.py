import whisperx
import torch
import os
from pydub import AudioSegment

# Setup
if torch.cuda.is_available():
    print("Using GPU for processing")
