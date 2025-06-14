import requests
import os
from pydub import AudioSegment
import time

# Config
INPUT_DIR = "polish_clips"
TRANSCRIPT_FILE = "transcripts.txt"
OUTPUT_DIR = "words_out"
TEMP_WAV_DIR = "temp_wavs"
GENTLE_URL = "http://localhost:8765/transcriptions?async=false"
MIN_WORD_DURATION = 0.5  # seconds


def align(audio_path, transcript):
    with open(audio_path, "rb") as audio_file:
        files = {"audio": audio_file}
        data = {"transcript": transcript}
        response = requests.post(GENTLE_URL, files=files, data=data)
        return response.json()


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


# Your helper function
def convert_to_wav(path, output_path):
    cmd = [
        "ffmpeg",
        "-i",
        path,
        "-ac",
        "1",
        "-ar",
        "24000",
        "-sample_fmt",
        "s16",
        output_path,
        "-y",
    ]
    subprocess.run(
        cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def main():
    start = time.time()

    transcripts = {}
    with open(TRANSCRIPT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if "\t" in line:
                filename, transcript = line.strip().split("\t", 1)
                transcripts[filename] = transcript

    os.makedirs(TEMP_WAV_DIR, exist_ok=True)

    for mp3_file in sorted(transcripts):
        if mp3_file != "common_voice_en_27608626.mp3":
            continue
        transcript = transcripts[mp3_file]
        mp3_path = os.path.join(INPUT_DIR, mp3_file)

        if not os.path.exists(mp3_path):

            print(f"⚠️ File not found: {mp3_file}")
            continue

        base = os.path.splitext(mp3_file)[0]
        wav_path = os.path.join(TEMP_WAV_DIR, base + ".wav")
        output_subdir = os.path.join(OUTPUT_DIR, base)

        print(f"🎧 Converting {mp3_file} to WAV...")
        try:
            convert_to_wav(mp3_path, wav_path)

            result = align(wav_path, transcript)

            segments = merge_short_words(result["words"])
            cut_audio_segments(wav_path, segments, output_subdir)

        except Exception as e:
            print(f"❌ Failed to process {mp3_file}: {e}\n")

    print(f"\n🕒 Done. Total time: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main()
