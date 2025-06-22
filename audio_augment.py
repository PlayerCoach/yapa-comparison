import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

sampling_rate = 16000
INPUT_DIR = "data/dataset/processed"
DESIRED_SET_COUNT = 800


def add_noise(audio, noise_level=0.005):
    noise = np.random.randn(len(audio))
    return audio + noise_level * noise


def change_pitch(audio, sr, n_steps=5):
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)


def change_speed(audio, speed_factor=1.2):
    return librosa.effects.time_stretch(audio, rate=speed_factor)


def apply_augmentations(audio, sr):
    return [
        ("noise", add_noise(audio)),
        ("pitch", change_pitch(audio, sr)),
        ("speed", change_speed(audio)),
    ]


def get_audio_files(directory):
    return [f for f in os.listdir(directory) if f.endswith(".wav")]


def augment_recursive(accent_dir, initial_count, desired_count):
    """
    Augments files in steps, creating new subfolders for each augmentation round.
    """
    round_idx = 1
    current_total = initial_count
    base_path = Path(accent_dir)
    round_input = base_path

    while current_total < desired_count:
        round_output = base_path / f"__augmented_{round_idx}"
        round_output.mkdir(exist_ok=True)

        files = get_audio_files(round_input)
        if not files:
            print(f"No files to augment in {round_input}, stopping.")
            break

        for file in files:
            if current_total >= desired_count:
                break

            file_path = round_input / file
            audio, sr = librosa.load(file_path, sr=sampling_rate)
            augmented_versions = apply_augmentations(audio, sr)

            for aug_type, aug_audio in augmented_versions:
                if current_total >= desired_count:
                    break
                new_filename = f"{file[:-4]}_{aug_type}.wav"
                sf.write(round_output / new_filename, aug_audio, sr)
                current_total += 1

        round_idx += 1
        round_input = round_output  # Next round uses previous augmentations


def get_accent_counts(input_root):
    """
    Returns dict of accent name -> number of wav files (including in augmented folders)
    """
    counts = dict()
    for accent in os.listdir(input_root):
        accent_path = os.path.join(input_root, accent)
        if not os.path.isdir(accent_path):
            continue

        total_count = 0
        for root, _, files in os.walk(accent_path):
            total_count += len([f for f in files if f.endswith(".wav")])
        counts[accent] = total_count
    return counts


if __name__ == "__main__":
    accent_counts = get_accent_counts(INPUT_DIR)

    for accent, count in accent_counts.items():
        accent_path = os.path.join(INPUT_DIR, accent)
        if not os.path.isdir(accent_path):
            continue

        if count < DESIRED_SET_COUNT:
            print(
                f"Augmenting accent '{accent}' from {count} to {DESIRED_SET_COUNT}..."
            )
            augment_recursive(accent_path, count, DESIRED_SET_COUNT)
        else:
            print(f"Accent '{accent}' already has {count} samples.")
