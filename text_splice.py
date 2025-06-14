import whisperx
import torch
import os
import subprocess
from pydub import AudioSegment
from contextlib import redirect_stdout, redirect_stderr

# Configuration
AUDIO_DIR = "polish_clips"
OUTPUT_DIR = "word_clips"
MIN_WORD_DURATION = 0.25  # in seconds


# Helpers
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
    subprocess.run(cmd, check=True)


def merge_short_words(words):
    merged = []
    i = 0
    while i < len(words):
        word = words[i]
        duration = word["end"] - word["start"]
        if duration >= MIN_WORD_DURATION:
            merged.append(
                {"word": word["word"], "start": word["start"], "end": word["end"]}
            )
            i += 1
        else:
            j = i + 1
            while (
                j < len(words)
                and (words[j]["end"] - words[j]["start"]) < MIN_WORD_DURATION
            ):
                j += 1
            if j < len(words):
                merged.append(
                    {
                        "word": f"{word['word']} {words[j]['word']}",
                        "start": word["start"],
                        "end": words[j]["end"],
                    }
                )
                i = j + 1
            else:
                merged.append(
                    {"word": word["word"], "start": word["start"], "end": word["end"]}
                )
                i += 1
    return merged


def process_audio_file(audio_path, model, align_model, align_metadata, device):
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    wav_path = os.path.join(AUDIO_DIR, f"{base_name}.wav")

    # Convert
    convert_to_wav(audio_path, wav_path)

    # Transcribe
    result = model.transcribe(wav_path)

    # Align
    aligned = whisperx.align(
        result["segments"], align_model, align_metadata, wav_path, device
    )

    # Merge short words
    word_segments = aligned.get("word_segments", [])
    merged_segments = merge_short_words(word_segments)

    # Slice audio
    audio = AudioSegment.from_wav(wav_path)
    output_path = os.path.join(OUTPUT_DIR, base_name)
    os.makedirs(output_path, exist_ok=True)

    for i, seg in enumerate(merged_segments):
        word = seg["word"].strip().replace(" ", "_").replace(".", "").replace(",", "")
        start = int(seg["start"] * 1000)
        end = int(seg["end"] * 1000)
        clip = audio[start:end]
        out_file = os.path.join(output_path, f"{i:03d}_{word}.wav")
        clip.export(out_file, format="wav")
        print(f"Exported: {out_file} ({seg['start']:.2f}s–{seg['end']:.2f}s)")

    # Clean up
    os.remove(wav_path)


# Main processing
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(os.devnull, "w") as fnull:
        with redirect_stdout(fnull), redirect_stderr(fnull):
            model = whisperx.load_model("large-v3", device)

    for filename in os.listdir(AUDIO_DIR):
        if not filename.lower().endswith((".mp3", ".wav", ".m4a", ".flac", ".ogg")):
            continue

        audio_path = os.path.join(AUDIO_DIR, filename)
        print(f"\nProcessing: {audio_path}")

        # Temporary convert to WAV to read language
        temp_wav = os.path.join(AUDIO_DIR, "temp.wav")
        convert_to_wav(audio_path, temp_wav)
        result = model.transcribe(temp_wav)
        os.remove(temp_wav)

        language = result["language"]
        align_model, align_metadata = whisperx.load_align_model(
            language_code=language, device=device
        )

        process_audio_file(audio_path, model, align_model, align_metadata, device)

    print("\n✅ All files processed.")


if __name__ == "__main__":
    main()
