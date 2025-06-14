import requests
import json
import os
from pydub import AudioSegment
import time

AUDIO_PATH = "books_synthesized.wav"
TRANSCRIPT = "Change doesn't happen all at once, it's slow, often invisible."
OUTPUT_DIR = "words_out"
GENTLE_URL = "http://localhost:8765/transcriptions?async=false"

MIN_WORD_DURATION = 0.25  # seconds, tweaked to reflect real short words


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
            else:
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


def main():
    print("Aligning...")
    result = align(AUDIO_PATH, TRANSCRIPT)
    print("Merging short words...")
    segments = merge_short_words(result["words"])
    print(f"Saving {len(segments)} segments...")
    cut_audio_segments(AUDIO_PATH, segments, OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"Total time: {end - start:.2f} seconds")


# TODO Przemnoz czas trwaniua przez to ile spowolnilem
