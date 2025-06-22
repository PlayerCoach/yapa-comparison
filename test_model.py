import pandas as pd
import numpy as np
from PIL import Image
from datasets import Dataset as HFDataset, Features, ClassLabel, Value
from sklearn.model_selection import train_test_split
import evaluate
from transformers import (
    ViTForImageClassification,
    ViTImageProcessor,
    TrainingArguments,  # type: ignore
    Trainer,  # type: ignore
)

import librosa
import torch
from transformers.trainer_callback import EarlyStoppingCallback  # type: ignore

import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import re
import tempfile
import soundfile as sf
import shutil


TSV_FILE = "data/cv-corpus-10.0-delta-2022-07-04/en/validated.tsv"
CLIPS_FOLDER = "data/cv-corpus-10.0-delta-2022-07-04/en/clips"

from audio_utils import convert_to_wav


def preprocess_audio(audio_file):
    from audio_utils import (
        convert_to_wav,
        normalize_audio,
        add_padding,
        denoise_wav,
        trim_silence,
    )

    normalize_audio(audio_file)
    denoise_wav(audio_file)
    trim_silence(audio_file)
    return audio_file


def filter_and_sort_tsv(accent, tsv_path=TSV_FILE, clips_path=CLIPS_FOLDER):
    used_clients = set()
    entries = []

    print(accent)
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if parts[0] == "client_id":
                continue

            client_id, filename, transcript, entry_accent = (
                parts[0],
                parts[1],
                parts[2],
                parts[7],
            )
            upvotes = int(parts[3]) if parts[3].isdigit() else 0
            downvotes = int(parts[4]) if parts[4].isdigit() else 0
            score = upvotes - downvotes

            if (
                accent == "Slavic"
            ):  # Special Case for slavic langagues, may delete later
                regex = re.compile(
                    r"\b(Slavic|Polish|Czech|Russian|Ukrainian|Bulgarian| \
                                Croatian|Slovak|Slovenian|Serbian|Latvian|Lithuanian| \
                                Hungarian|Romanian|Kazakh|Azerbaijani|Georgian|Moldovan)\b",
                    re.IGNORECASE,
                )
            else:
                regex = re.compile(rf"^{re.escape(accent)}$", re.IGNORECASE)

            # if client_id in used_clients:
            #     continue
            if not regex.search(entry_accent):
                continue

            audio_path = os.path.join(clips_path, filename)
            if not os.path.exists(audio_path):
                continue
            if sf.info(audio_path).duration < 5.0:
                continue  # duration is in seconds, require at least 3 seconds

            used_clients.add(client_id)
            entries.append((score, filename, transcript))

    entries.sort(key=lambda x: (x[0] != 0, -x[0]))

    temp = tempfile.NamedTemporaryFile(
        delete=False, mode="w", encoding="utf-8", suffix=".tsv"
    )
    for score, filename, transcript in entries:
        temp.write(f"{filename}\t{transcript}\n")
    temp.close()
    print(f"Filtered TSV saved to: {temp.name}")
    return temp.name


def create_spectogram(audio_path):
    # Convert to wav in a temp file
    with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as wav_temp:
        convert_to_wav(audio_path, wav_temp.name)
        wav_temp.name = preprocess_audio(wav_temp.name)
        y, sr = librosa.load(wav_temp.name, sr=16000)

        if len(y) < 512:
            print("warning: short audio")

        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=128, n_fft=512, hop_length=128
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as img_temp:
            fig, ax = plt.subplots()
            librosa.display.specshow(mel_db, sr=sr, ax=ax)
            ax.axis("off")
            plt.savefig(img_temp.name, bbox_inches="tight", pad_inches=0)
            plt.close(fig)

            img = Image.open(img_temp.name).convert("RGB").resize((244, 244))
            img = (np.array(img).astype(np.float32) / 255.0 * 255).astype(np.uint8)
            return Image.fromarray(img).resize((224, 224)).convert("RGB")


def predict(model, processor, image):
    # Inference
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.nn.functional.softmax(logits, dim=-1)
    pred = int(torch.argmax(probs, dim=-1).item())
    return model.config.id2label[pred]


import shutil

# Add this near the top
MISCLASSIFIED_DIR = "misclassified_audios"
os.makedirs(MISCLASSIFIED_DIR, exist_ok=True)

if __name__ == "__main__":
    # Load model
    model = ViTForImageClassification.from_pretrained(
        "./yapa_comparission/checkpoint-900"
    )
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    ACCENTS = [
        "Australian English",
        "Canadian English",
        "England English",
        "India and South Asia (India, Pakistan, Sri Lanka)",
        "Irish English",
        "Scottish English",
        "United States English",
        "Filipino",
        "Slavic",  # Special case
    ]

    ACCENT_LABEL_MAP = {
        "Australian English": "australian",
        "Canadian English": "canadian",
        "England English": "england",
        "India and South Asia (India, Pakistan, Sri Lanka)": "india",
        "Irish English": "irish",
        "Scottish English": "scottish",
        "United States English": "american",
        "Filipino": "filipino",
        "Slavic": "slavic",
    }
    total_hits = 0
    total_predictions = 0

    misclassified_count = 0

    for accent in ACCENTS:
        hits = 0
        predictions = 0
        temp_tsv = filter_and_sort_tsv(accent)
        with open(temp_tsv, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                filename = parts[0]
                audio_path = os.path.join(CLIPS_FOLDER, filename)
                image = create_spectogram(audio_path)
                prediction = predict(model, processor, image).strip().lower()
                real_label = ACCENT_LABEL_MAP[accent].strip().lower()

                if real_label == prediction:
                    hits += 1
                    total_hits += 1
                else:
                    misclassified_count += 1
                    if misclassified_count % 25 == 0:
                        # Convert to wav if not already
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".wav"
                        ) as temp_wav:
                            convert_to_wav(audio_path, temp_wav.name)
                            dest_filename = f"{real_label}_{prediction}_{os.path.splitext(filename)[0]}.wav"
                            dest_path = os.path.join(MISCLASSIFIED_DIR, dest_filename)
                            shutil.copy(temp_wav.name, dest_path)

                total_predictions += 1
                predictions += 1

        os.remove(temp_tsv)
        if predictions == 0:
            print(f"No data for label {accent}")
        else:
            print(f"{accent} accuracy: {hits/predictions}")
            print(f"Samples: {predictions}")

    if total_predictions == 0:
        print("No data for labels")
    else:
        print(f"Total accuracy of the model: {total_hits/total_predictions}")
        print(f"Total predictions: {total_predictions}")
