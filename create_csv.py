import os
import pandas as pd
from collections import defaultdict
import random

spectrogram_dir = "data/new_dataset/spectrograms"


def create_csv():
    """
    Creates a balanced CSV file with equal American and British samples.
    Skips 'other' labels and ensures equal representation.
    """
    data_by_label = defaultdict(list)

    # Collect all image paths by label
    for fname in os.listdir(spectrogram_dir):
        if fname.endswith(".png"):
            label = fname.split("_")[0].lower()
            if label not in {"american", "british"}:
                continue

            image_path = os.path.join(spectrogram_dir, fname)
            data_by_label[label].append((image_path, label))

    # Trim to equal count
    min_count = min(len(data_by_label["american"]), len(data_by_label["british"]))
    print(f"🧪 Using {min_count} samples per class")

    balanced_data = random.sample(data_by_label["american"], min_count) + random.sample(
        data_by_label["british"], min_count
    )

    random.shuffle(balanced_data)
    df = pd.DataFrame(balanced_data, columns=["image_path", "label"])
    df.to_csv("spectrogram_dataset.csv", index=False)

    print(f"✅ Saved balanced spectrogram_dataset.csv with {len(df)} entries.")
    print(f"British entries: {min_count}")
    print(f"American entries: {min_count}")
    print("CSV file created successfully.")


if __name__ == "__main__":
    create_csv()
