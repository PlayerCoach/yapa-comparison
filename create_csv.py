import os
import pandas as pd
from collections import defaultdict
import random

spectrogram_dir = "data/dataset/spectrograms"


def create_csv():
    """
    Creates a balanced CSV file from spectrograms by:
    - Automatically detecting all accent classes from filenames
    - Scanning both 'train' and 'test' subfolders
    - Balancing all classes (per split) to within ±10% of the smallest class
    - Writing the result to a CSV with 'split' column
    - Printing original and final counts
    """
    data_by_split_and_label = defaultdict(lambda: defaultdict(list))

    # Step 1: Walk through spectrogram_dir and group by split and label
    for root, _, files in os.walk(spectrogram_dir):
        for fname in files:
            if not fname.endswith(".png"):
                continue

            # Determine split from folder name
            rel_path = os.path.relpath(root, spectrogram_dir)
            split = rel_path.split(os.sep)[0].lower()
            if split not in {"train", "test"}:
                continue

            label = fname.split("_")[0].lower()
            image_path = os.path.join(root, fname)
            data_by_split_and_label[split][label].append((image_path, label, split))

    # Step 2: Show original distribution
    print("📊 Original class counts by split:")
    for split in data_by_split_and_label:
        print(f"\n🔹 {split.upper()}:")
        for label, samples in data_by_split_and_label[split].items():
            print(f"  - {label}: {len(samples)} samples")

    all_balanced_data = []

    for split, label_dict in data_by_split_and_label.items():
        # Step 3: Determine min class size for this split
        class_counts = {label: len(samples) for label, samples in label_dict.items()}
        min_count = min(class_counts.values())
        min_allowed = int(min_count * 0.9)
        max_allowed = int(min_count * 1.1)

        print(f"\n📉 Balancing {split.upper()} to [{min_allowed}, {max_allowed}]")

        for label, samples in label_dict.items():
            count = len(samples)
            if count < min_allowed:
                print(
                    f"⚠️ Skipping {split}/{label}: only {count} samples (< {min_allowed})"
                )
                continue
            target_count = min(count, max_allowed)
            selected = random.sample(samples, target_count)
            all_balanced_data.extend(selected)

    # Step 4: Save combined CSV
    random.shuffle(all_balanced_data)
    df = pd.DataFrame(all_balanced_data, columns=["image_path", "label", "split"])
    df.to_csv("spectrogram_dataset.csv", index=False)

    print(f"\n✅ Saved spectrogram_dataset.csv with {len(df)} total entries.")

    # Step 5: Final class count summary
    print("\n📦 Final balanced class counts by split:")
    for split in {"train", "test"}:
        sub_df = df[df["split"] == split]
        print(f"\n🔹 {split.upper()}:")
        for label in sub_df["label"].unique():
            count = (sub_df["label"] == label).sum()
            print(f"  - {label}: {count} samples")


if __name__ == "__main__":
    create_csv()
