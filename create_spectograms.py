import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Paths
processed_audio_path = "data/dataset/processed"
output_root = "data/dataset/spectrograms"
img_size = (244, 244)
data = []


def check_if_spectrograms_exist(image_path: str) -> bool:
    return os.path.exists(image_path)


def create_spectrograms_recursive():
    for split in ["train", "test"]:
        split_audio_path = os.path.join(processed_audio_path, split)
        split_output_path = os.path.join(output_root, split)
        os.makedirs(split_output_path, exist_ok=True)

        for accent in os.listdir(split_audio_path):
            accent_audio_path = os.path.join(split_audio_path, accent)

            for root, _, files in os.walk(accent_audio_path):
                for fname in files:
                    if not fname.endswith(".wav"):
                        continue

                    rel_path = os.path.relpath(
                        os.path.join(root, fname), accent_audio_path
                    )
                    rel_part = rel_path.replace(os.sep, "_").replace(".wav", "")
                    image_fname = f"{accent}_{rel_part}.png"
                    image_path = os.path.join(split_output_path, image_fname)

                    if check_if_spectrograms_exist(image_path):
                        print(f"✅ {image_fname} exists, skipping.")
                        continue

                    wav_path = os.path.join(root, fname)
                    y, sr = librosa.load(wav_path, sr=16000)

                    if len(y) < 512:
                        print(f"⚠️ Skipping short audio: {image_fname}")
                        continue

                    mel = librosa.feature.melspectrogram(
                        y=y, sr=sr, n_mels=128, n_fft=512, hop_length=128
                    )
                    mel_db = librosa.power_to_db(mel, ref=np.max)

                    fig, ax = plt.subplots()
                    librosa.display.specshow(mel_db, sr=sr, ax=ax)
                    ax.axis("off")
                    plt.savefig(image_path, bbox_inches="tight", pad_inches=0)
                    plt.close(fig)

                    img = Image.open(image_path).convert("RGB").resize(img_size)
                    img = np.array(img).astype(np.float32) / 255.0
                    img = (img * 255).astype(np.uint8)
                    Image.fromarray(img).save(image_path)

                    data.append((image_path, accent))


if __name__ == "__main__":
    create_spectrograms_recursive()
    print("✅ Spectrograms created successfully.")
    print(f"Total spectrograms created: {len(data)}")
