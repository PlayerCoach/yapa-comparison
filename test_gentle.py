import requests
import os
from pydub import AudioSegment
import time


TEMP_WAV_DIR = "temp_wavs"
GENTLE_URL = "http://localhost:8765/transcriptions?async=false"


def align(audio_path, transcript):
    with open(audio_path, "rb") as audio_file:
        files = {"audio": audio_file}
        data = {"transcript": transcript}
        response = requests.post(GENTLE_URL, files=files, data=data)
        return response.json()


def is_aligned(result):
    # Gentle returns a "words" list with alignment info
    words = result.get("words", [])
    for w in words:
        # If a word is not aligned, "start" or "end" might be missing or None
        if w.get("case") != "success":
            return False
    return True


if __name__ == "__main__":
    audio_path = "samples/books.m4a"  # Replace with your actual audio file path
    transcript = "Change doesn't happen all at once, it's slow, often invisible."

    alignment_result = align(audio_path, transcript)
    print(is_aligned(alignment_result))
