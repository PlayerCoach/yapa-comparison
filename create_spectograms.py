import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import csv

# Paths
processed_audio_path = "data/new_dataset/processed_audio"
output_dir = "data/new_dataset/spectrograms"
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


def create_spectrograms():
    """
    Converts WAV files into mel spectrograms, saves them as normalized 3-channel images,
    and logs their paths and labels.
    """
    accent_folders = os.listdir(processed_audio_path)

    for accent in accent_folders:
        accent_path = os.path.join(processed_audio_path, accent)
        for sentence in os.listdir(accent_path):
            sentence_path = os.path.join(accent_path, sentence)
            for fname in os.listdir(sentence_path):
                if not fname.endswith(".wav"):
                    continue
                if check_if_spectrograms_exist(
                    accent + "_" + sentence + "_" + fname.replace(".wav", ".png")
                ):
                    print(
                        f"Spectrogram for {accent}_{sentence}_{fname} already exists, skipping."
                    )
                    continue
                wav_path = os.path.join(sentence_path, fname)
                y, sr = librosa.load(wav_path, sr=16000)  # Load audio file at 24kHz

                # Generate mel spectrogram
                mel = librosa.feature.melspectrogram(
                    y=y, sr=sr, n_mels=128, n_fft=512, hop_length=128
                )
                mel_db = librosa.power_to_db(mel, ref=np.max(mel))

                # Plot without axes
                fig, ax = plt.subplots()
                librosa.display.specshow(mel_db, sr=sr, x_axis=None, y_axis=None, ax=ax)
                ax.axis("off")
                image_fname = f"{accent}_{sentence}_{fname.replace('.wav', '.png')}"
                image_path = os.path.join(output_dir, image_fname)
                plt.savefig(image_path, bbox_inches="tight", pad_inches=0)
                plt.close(fig)

                # Resize, convert to RGB, normalize
                img = Image.open(image_path).convert("RGB").resize(img_size)
                img = np.array(img).astype(np.float32) / 255.0  # Normalize to [0, 1]
                img = (img * 255).astype(np.uint8)  # Convert back for saving
                Image.fromarray(img).save(image_path)

                data.append((image_path, accent))


def create_spectrograms_secondary():
    """Created spectograms for alternative folder structure processe_audio/accent/ .wav"""
    accent_folders = os.listdir(processed_audio_path)
    for accent in accent_folders:
        accent_path = os.path.join(processed_audio_path, accent)
        for fname in os.listdir(accent_path):
            if not fname.endswith(".wav"):
                continue
            if check_if_spectrograms_exist(
                accent + "_" + fname.replace(".wav", ".png")
            ):
                print(f"Spectrogram for {accent}_{fname} already exists, skipping.")
                continue
            wav_path = os.path.join(accent_path, fname)
            y, sr = librosa.load(wav_path, sr=16000)  # Load audio file at 16kHz

            if len(y) < 512:
                # print(f"❌ Audio file {fname} is too short for mel spectrogram.")
                continue
            # Generate mel spectrogram
            mel = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=128, n_fft=512, hop_length=128
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)

            # Plot without axes
            fig, ax = plt.subplots()
            librosa.display.specshow(mel_db, sr=sr, x_axis=None, y_axis=None, ax=ax)
            ax.axis("off")
            image_fname = f"{accent}_{fname.replace('.wav', '.png')}"
            image_path = os.path.join(output_dir, image_fname)
            plt.savefig(image_path, bbox_inches="tight", pad_inches=0)
            plt.close(fig)

            # Resize, convert to RGB, normalize
            img = Image.open(image_path).convert("RGB").resize(img_size)
            img = np.array(img).astype(np.float32) / 255.0  # Normalize to [0, 1]
            img = (img * 255).astype(np.uint8)  # Convert back for saving
            Image.fromarray(img).save(image_path)

            data.append((image_path, accent))


if __name__ == "__main__":
    create_spectrograms_secondary()

    print("✅ Spectrograms created successfully.")
    print(f"Total spectrograms created: {len(data)}")
    print(f"Spectrograms saved in: {output_dir}")
