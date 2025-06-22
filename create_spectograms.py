import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import csv

# Paths
processed_audio_path = "data/dataset/processed"
output_dir = "data/dataset/spectrograms"
os.makedirs(output_dir, exist_ok=True)

# Spectrogram image size for ViT
img_size = (244, 244)

# Store (image_path, label) pairs
data = []


def check_if_spectrograms_exist(wav_name: str) -> bool:
    """
    Check if the spectrograms CSV file already exists.
    If it exits, then do nothing, it is in the correct folder
    """

    if os.path.exists(os.path.join(output_dir, wav_name)):
        return True
    else:
        return False


def create_spectrograms_recursive():
    """
    Recursively traverses processed_audio_path to find all WAV files,
    generates spectrograms, and stores (image_path, label) pairs.
    Ensures each image filename starts with the accent name.
    """
    for accent_folder in os.listdir(processed_audio_path):
        accent_path = os.path.join(processed_audio_path, accent_folder)
        if not os.path.isdir(accent_path):
            continue

        # Extract accent from folder name, e.g. "american" from "american_dataset"
        accent = accent_folder.replace("_dataset", "")

        for root, _, files in os.walk(accent_path):
            for fname in files:
                if not fname.endswith(".wav"):
                    continue

                # Construct image filename: accent + rest of relative path
                relative_path = os.path.relpath(os.path.join(root, fname), accent_path)
                relative_part = relative_path.replace(os.sep, "_").replace(".wav", "")
                image_fname = f"{accent}_{relative_part}.png"

                if check_if_spectrograms_exist(image_fname):
                    print(f"✅ Spectrogram for {image_fname} already exists, skipping.")
                    continue

                wav_path = os.path.join(root, fname)
                y, sr = librosa.load(wav_path, sr=16000)

                if len(y) < 512:
                    print(f"⚠️ Skipping short audio: {image_fname}")
                    continue

                # Generate mel spectrogram
                mel = librosa.feature.melspectrogram(
                    y=y, sr=sr, n_mels=128, n_fft=512, hop_length=128
                )
                mel_db = librosa.power_to_db(mel, ref=np.max)

                # Plot and save image
                fig, ax = plt.subplots()
                librosa.display.specshow(mel_db, sr=sr, ax=ax)
                ax.axis("off")
                image_path = os.path.join(output_dir, image_fname)
                plt.savefig(image_path, bbox_inches="tight", pad_inches=0)
                plt.close(fig)

                # Resize, convert to RGB, normalize
                img = Image.open(image_path).convert("RGB").resize(img_size)
                img = np.array(img).astype(np.float32) / 255.0
                img = (img * 255).astype(np.uint8)
                Image.fromarray(img).save(image_path)

                data.append((image_path, accent))


if __name__ == "__main__":
    create_spectrograms_recursive()
    print("✅ Spectrograms created successfully.")
    print(f"Total spectrograms created: {len(data)}")
    print(f"Spectrograms saved in: {output_dir}")
