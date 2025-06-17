from pick_audio import pick_audio
from text_splice import splice_audio_files
import os

ACCENT_FOLDERS = [
    "british",
    "american",
    "other",
]

if __name__ == "__main__":
    pick_audio(size=2000, output="data/dataset", input="big_data")
    for accent in ACCENT_FOLDERS:
        input_dir = os.path.join("data/dataset", accent + "_clips")
        output_dir = os.path.join("data/dataset/processed_audio", accent)
        transcript_file = os.path.join(input_dir, accent + ".tsv")
        splice_audio_files(
            INPUT_DIR=input_dir, TRANSCRIPT_FILE=transcript_file, OUTPUT_DIR=output_dir
        )

    from create_spectograms import create_spectrograms

    create_spectrograms()
    print("✅ Spectrograms created successfully.")
