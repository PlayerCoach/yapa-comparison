import os
import pandas as pd

spectrogram_dir = "spectrograms"
data = []

for fname in os.listdir(spectrogram_dir):
    if fname.endswith(".png"):
        label = fname.split("_")[0]  # assumes label is before first underscore
        image_path = os.path.join(spectrogram_dir, fname)
        data.append((image_path, label))

# Save as CSV
df = pd.DataFrame(data, columns=["image_path", "label"])
df.to_csv("spectrogram_dataset.csv", index=False)

print("✅ Saved spectrogram_dataset.csv with", len(df), "entries.")
