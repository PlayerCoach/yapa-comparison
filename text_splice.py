import whisperx
import torch
import os
from pydub import AudioSegment

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisperx.load_model("large-v3", device)

# Transcribe
audio_path = "polish_clips/common_voice_en_21916640.mp3"
result = model.transcribe(audio_path)
print(result)

# Alignment
model_a, metadata = whisperx.load_align_model(
    language_code=result["language"], device=device
)
aligned = whisperx.align(
    result["segments"],
    model_a,
    metadata,
    audio_path,
    device,
)

# Word-level audio slicing
audio = AudioSegment.from_wav(audio_path)
os.makedirs("word_clips", exist_ok=True)

for i, word_info in enumerate(aligned.get("word_segments", [])):
    word = (
        word_info.get("text", f"unk_{i}")
        .strip()
        .replace(" ", "_")
        .replace(".", "")
        .replace(",", "")
    )
    start = int(word_info["start"] * 1000)
    end = int(word_info["end"] * 1000)
    word_audio = audio[start:end]
    file_path = os.path.join("word_clips", f"{i:03d}_{word}.wav")
    word_audio.export(file_path, format="wav")
