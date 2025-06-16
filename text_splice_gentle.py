import requests
import os
from pydub import AudioSegment
import time

# Config
INPUT_DIR = "other_clips"
TRANSCRIPT_FILE = "other_english_subset.txt"
OUTPUT_DIR = "processed_audio/other_clips_words"
TEMP_WAV_DIR = "temp_wavs"
GENTLE_URL = "http://localhost:8765/transcriptions?async=false"
MIN_WORD_DURATION = 0.5  # seconds


def align(audio_path, transcript):
    with open(audio_path, "rb") as audio_file:
        files = {"audio": audio_file}
        data = {"transcript": transcript}
        response = requests.post(GENTLE_URL, files=files, data=data)
        return response.json()


def denoise_results(path):
    """
    Denoise all .wav files in the given directory and subdirectories.
    """
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".wav"):
                full_path = os.path.join(root, file)
                try:
                    denoise_wav(full_path)
                except Exception as e:
                    print(f"❌ Failed to denoise {full_path}: {e}")


def merge_short_words(words):
    merged = []
    i = 0
    while i < len(words):
        word = words[i]
        if word.get("case") != "success":
            i += 1
            continue

        duration = word["end"] - word["start"]
        if duration >= MIN_WORD_DURATION:
            merged.append(
                {"word": word["word"], "start": word["start"], "end": word["end"]}
            )
            i += 1
        else:
            # Try to merge with next successful word
            j = i + 1
            while j < len(words) and words[j].get("case") != "success":
                j += 1

            if j < len(words):
                next_word = words[j]
                merged.append(
                    {
                        "word": f"{word['word']} {next_word['word']}",
                        "start": word["start"],
                        "end": next_word["end"],
                    }
                )
                i = j + 1
            elif merged:
                # No next word; merge with previous
                merged[-1]["word"] += f" {word['word']}"
                merged[-1]["end"] = word["end"]
                i += 1
            else:
                # No previous either, just add it
                merged.append(
                    {"word": word["word"], "start": word["start"], "end": word["end"]}
                )
                i += 1
    return merged


def cut_audio_segments(audio_path, segments, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    audio = AudioSegment.from_wav(audio_path)
    for i, seg in enumerate(segments):
        start_ms = int(seg["start"] * 1000)
        end_ms = int(seg["end"] * 1000)
        clip = audio[start_ms:end_ms]
        filename = f"{i:03d}_{seg['word'].replace(' ', '_')}.wav"
        clip.export(os.path.join(output_dir, filename), format="wav")


import subprocess

from audio_utils import *


# Your helper function
def preprocess_audio(path, output_path):
    convert_to_wav(path, output_path)
    # denoise_wav(output_path)
    normalize_audio(output_path)
    add_padding(output_path)
    return output_path


def main(MAX_FILES=1000):
    start = time.time()
    counter = 0
    transcripts = {}
    with open(TRANSCRIPT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if "\t" in line:
                filename, transcript = line.strip().split("\t", 1)
                transcripts[filename] = transcript

    os.makedirs(TEMP_WAV_DIR, exist_ok=True)

    for mp3_file in sorted(transcripts):
        if counter >= MAX_FILES:
            break

        transcript = transcripts[mp3_file]
        mp3_path = os.path.join(INPUT_DIR, mp3_file)

        if not os.path.exists(mp3_path):

            print(f"⚠️ File not found: {mp3_file}")
            continue

        base = os.path.splitext(mp3_file)[0]
        wav_path = os.path.join(TEMP_WAV_DIR, base + ".wav")
        output_subdir = os.path.join(OUTPUT_DIR, base)

        try:
            preprocess_audio(mp3_path, wav_path)

            result = align(wav_path, transcript)

            segments = merge_short_words(result["words"])
            cut_audio_segments(wav_path, segments, output_subdir)
            denoise_results(output_subdir)
            print(
                f"✅ Processed {mp3_file} successfully. Segments saved to {output_subdir}"
            )
            # deno
            counter += 1

        except Exception as e:
            print(f"❌ Failed to process {mp3_file}: {e}\n")

    print(f"\n🕒 Done. Total time: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main(100)  # Adjust MAX_FILES as needed
    # delete tempwavs dir
    if os.path.exists(TEMP_WAV_DIR):
        subprocess.run(["rm", "-rf", TEMP_WAV_DIR])
        print(f"🗑️ Temporary directory {TEMP_WAV_DIR} deleted.")
    else:
        print(f"⚠️ Temporary directory {TEMP_WAV_DIR} does not exist.")
