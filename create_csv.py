import os
import pandas as pd

spectrogram_dir = "data/new_dataset/spectrograms"
data = []


def create_csv():
    """
    Creates a CSV file from spectrogram images and their labels.
    Skips images labeled as 'other'.
    """

    for fname in os.listdir(spectrogram_dir):
        if fname.endswith(".png"):
            label = fname.split("_")[0]  # assumes label is before first underscore
            if label == "other":
                continue  # Skip 'other' label
            image_path = os.path.join(spectrogram_dir, fname)
            data.append((image_path, label))

    # Save as CSV
    df = pd.DataFrame(data, columns=["image_path", "label"])
    df.to_csv("spectrogram_dataset.csv", index=False)

    print("✅ Saved spectrogram_dataset.csv with", len(df), "entries.")
