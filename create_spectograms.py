import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

accent_folders = os.listdir("processed_audio")
output_dir = "spectrograms"
os.makedirs(output_dir, exist_ok=True)

img_size = (224, 224)  # Standard ViT input size

data = []  # Store (image_path, label) pairs

for accent in accent_folders:
    accent_path = os.path.join("processed_audio", accent)
    for sentence in os.listdir(accent_path):
        sentence_path = os.path.join(accent_path, sentence)
        for fname in os.listdir(sentence_path):
            if not fname.endswith(".wav"):
                continue
            wav_path = os.path.join(sentence_path, fname)
            y, sr = librosa.load(wav_path, sr=None)

            # if len(y) < sr * 0.2:  # skip files shorter than 200ms
            #     print(f"[SKIP] Too short: {wav_path}")
            #     continue

            # Generate mel spectrogram
            mel = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=128, n_fft=512, hop_length=128
            )

            mel_db = librosa.power_to_db(mel, ref=np.max)

            # Plot and save as image
            fig, ax = plt.subplots()
            librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", ax=ax)
            ax.axis("off")
            image_fname = f"{accent}_{sentence}_{fname.replace('.wav', '.png')}"
            image_path = os.path.join(output_dir, image_fname)
            plt.savefig(image_path, bbox_inches="tight", pad_inches=0)
            plt.close(fig)

            # Resize and convert to 3-channel (ViT expects RGB)
            img = Image.open(image_path).convert("RGB").resize(img_size)
            img.save(image_path)

            data.append((image_path, accent))
