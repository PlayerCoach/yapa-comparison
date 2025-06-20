from audio_utils import *
import os


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

    for root, _, files in os.walk(INPUT_DIR):
        for file in sorted(files):
            if not file.endswith(".mp3"):
                continue

            rel_dir = os.path.relpath(root, INPUT_DIR)
            out_dir = os.path.join(OUTPUT_DIR, rel_dir)
            os.makedirs(out_dir, exist_ok=True)

            input_path = os.path.join(root, file)
            base = os.path.splitext(file)[0]
            output_path = os.path.join(out_dir, base + ".wav")

            try:
                preprocess_audio(input_path, output_path)
                # print(f"✅ Processed {input_path}")
            except Exception as e:
                print(f"❌ Failed to process {input_path}: {e}")
