import os
import pandas as pd
from collections import defaultdict
import random

spectrogram_dir = "data/dataset/spectrograms"


def create_csv():
    """
    Creates a balanced CSV file from spectrograms by:
    - Automatically detecting all accent classes from filenames
    - Balancing all classes to within ±10% of the smallest class
    - Writing the result to a CSV
    - Printing original and final counts
    """
    data_by_label = defaultdict(list)

    # Step 1: Group spectrograms by label (prefix of filename)
    for fname in os.listdir(spectrogram_dir):
        if not fname.endswith(".png"):
            continue

        label = fname.split("_")[0].lower()
        image_path = os.path.join(spectrogram_dir, fname)
        data_by_label[label].append((image_path, label))

    # Step 2: Show original class distribution
    all_class_counts = {label: len(samples) for label, samples in data_by_label.items()}
    print("📊 Original class counts:")
    for label, count in all_class_counts.items():
        print(f"  - {label}: {count} samples")

    # Step 3: Determine min class size and allowed range
    min_count = min(all_class_counts.values())
    min_allowed = int(min_count * 0.9)
    max_allowed = int(min_count * 1.1)

    print(
        f"\n📉 Balancing to min class count ±10%: range [{min_allowed}, {max_allowed}]"
    )

    # Step 4: Balance dataset
    balanced_data = []
    final_counts = {}

    for label, samples in data_by_label.items():
        sample_count = len(samples)

        if sample_count < min_allowed:
            print(f"⚠️ Skipping {label}: only {sample_count} samples (< {min_allowed})")
            continue

        target_count = min(sample_count, max_allowed)
        selected_samples = random.sample(samples, target_count)
        balanced_data.extend(selected_samples)
        final_counts[label] = target_count

    # Step 5: Shuffle and save
    random.shuffle(balanced_data)
    df = pd.DataFrame(balanced_data, columns=["image_path", "label"])
    df.to_csv("spectrogram_dataset.csv", index=False)

    print(f"\n✅ Saved spectrogram_dataset.csv with {len(df)} total entries.")

    # Step 6: Show final balanced class counts
    print("📦 Final balanced class counts:")
    for label, count in final_counts.items():
        print(f"  - {label}: {count} samples")


if __name__ == "__main__":
    create_csv()
