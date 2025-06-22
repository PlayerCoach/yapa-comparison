from audio_utils import *
import os
import random


def preprocess_audio(path, output_path):
    from audio_utils import convert_to_wav, normalize_audio, add_padding

    convert_to_wav(path, output_path)
    normalize_audio(output_path)
    denoise_wav(output_path)
    trim_silence(output_path)
    return output_path


def batch_process_audio(INPUT_DIR=None, OUTPUT_DIR=None):
    if INPUT_DIR is None:
        raise ValueError("INPUT_DIR must be specified")
    if OUTPUT_DIR is None:
        raise ValueError("OUTPUT_DIR must be specified")

    for accent in os.listdir(INPUT_DIR):
        accent_path = os.path.join(INPUT_DIR, accent)
        if not os.path.isdir(accent_path):
            continue

        all_files = sorted([f for f in os.listdir(accent_path) if f.endswith(".mp3")])
        random.shuffle(all_files)

        split_idx = int(0.8 * len(all_files))
        train_files = all_files[:split_idx]
        test_files = all_files[split_idx:]

        for subset, files in zip(["train", "test"], [train_files, test_files]):
            for file in files:
                input_path = os.path.join(accent_path, file)
                out_dir = os.path.join(OUTPUT_DIR, subset, accent)
                os.makedirs(out_dir, exist_ok=True)
                base = os.path.splitext(file)[0]
                output_path = os.path.join(out_dir, base + ".wav")

                try:
                    preprocess_audio(input_path, output_path)
                except Exception as e:
                    print(f"❌ Failed to process {input_path}: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_dir",
        type=str,
        default="data/dataset",
        help="Input folder where you have folders with labeled accents",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/dataset/processed",
        help="Output folder where the processed audio files will be stored",
    )
    args = parser.parse_args()
    batch_process_audio(args.in_dir, args.out_dir)
