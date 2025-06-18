from pick_audio import pick_audio
from text_splice import splice_audio_files
from machine import Machine
from create_spectograms import create_spectrograms
from create_csv import create_csv
from push_to_hugging import push_to_huggingface
import os

ACCENT_FOLDERS = [
    "british",
    "american",
]

if __name__ == "__main__":
    pick_audio(size=10, output="data/new_dataset", input="big_data")
    for accent in ACCENT_FOLDERS:
        input_dir = os.path.join("data/new_dataset", accent + "_clips")
        output_dir = os.path.join("data/new_dataset/processed_audio", accent)
        transcript_file = os.path.join(input_dir, accent + ".tsv")
        splice_audio_files(
            INPUT_DIR=input_dir, TRANSCRIPT_FILE=transcript_file, OUTPUT_DIR=output_dir
        )

    from create_spectograms import create_spectrograms

    create_spectrograms()
    create_csv()
    machine = Machine()
    machine.learn()
    machine.evaluate()
    push_to_huggingface()

    print("✅ It's either working or im done with this project.")
