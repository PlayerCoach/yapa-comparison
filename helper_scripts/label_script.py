"""
Script that helps to label data.

Run the script in VSCODE in the yapa_comparison folder!, label in the file manager, manage script in the terminal below.

Specify the accent that you want to label in the `accent_regex` variable. (You can find all accent in the accent.txt with how
many samples are in each accent.)

Specify the number of samples you want to label in the `size` variable.

Specify the path to the CSV file with labeled data in the `csv_path` variable.

Specify the folder where the audio files are located in the `audio_dir` variable.
In current directory folder will be created, in this folder there there will be .cvs file with labled data, all labled audio and one folder
with the audio to label.

Press 'Enter' to move audio from the audio_to_label to main folder with audio and update the .csv file.
Press 'q' to quit the script. It will delte the folder with audio to label.

The script will remember the user id so that all samples that you get to label are from someone you haven't labeled before.
Script will also sort out too short audio.
"""

ACCENT_LABEL = "England English"
CSV_PATH = "data/cv-corpus-21.0-2025-03-14/en/validated.tsv"
AUDIO_DIR = "data/cv-corpus-21.0-2025-03-14/en/clips"
SIZE = 200
BATCH_SIZE = 10

OUTPUT_FOLDER_NAME = "Labeled_data"

import os
import re
import tempfile
import soundfile as sf
import shutil

count_files = lambda path: len(
    [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
)


def filter_and_sort_tsv(tsv_path=CSV_PATH, clips_path=AUDIO_DIR, min_frames=1024):
    used_clients = set()
    entries = []

    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 10 or parts[0] == "client_id":
                continue

            client_id, filename, transcript, accent = (
                parts[0],
                parts[1],
                parts[3],
                parts[9],
            )
            upvotes = int(parts[5]) if parts[5].isdigit() else 0
            downvotes = int(parts[6]) if parts[6].isdigit() else 0
            score = upvotes - downvotes

            if (
                ACCENT_LABEL == "Slavic"
            ):  # Special Case for slavic langagues, may delete later
                regex = re.compile(
                    r"\b(Slavic|Polish|Czech|Russian|Ukrainian|Bulgarian| \
                                Croatian|Slovak|Slovenian|Serbian|Latvian|Lithuanian| \
                                Hungarian|Romanian|Kazakh|Azerbaijani|Georgian|Moldovan)\b",
                    re.IGNORECASE,
                )
            else:
                regex = re.compile(rf"^{re.escape(ACCENT_LABEL)}$", re.IGNORECASE)

            if client_id in used_clients:
                continue
            if not regex.search(accent):
                continue

            audio_path = os.path.join(clips_path, filename)
            if not os.path.exists(audio_path):
                continue
            if sf.info(audio_path).frames < min_frames:
                continue

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


def get_next_batch(temp_tsv, index) -> bool:
    EOF = False
    new_batch = []

    batch_dir = os.path.join(OUTPUT_FOLDER_NAME, "Data_to_label")

    with open(temp_tsv, "r", encoding="utf-8") as f:
        for _ in range(index):
            next(f, None)
        for _ in range(BATCH_SIZE):
            line = f.readline()
            if not line:
                EOF = True
                break
            parts = line.strip().split("\t")
            filename = parts[0]
            new_batch.append(filename)
            src_file = os.path.join(AUDIO_DIR, filename)
            dst_file = os.path.join(batch_dir, filename)
            if os.path.isfile(src_file):
                shutil.copy2(src_file, dst_file)

    print(f"Prepared {len(new_batch)} files in batch.")
    if len(new_batch) == 0:
        print(
            "No files that specify criteria were found, check if choosen accent is in the dataset"
        )

    return EOF


def main():
    batch_dir = os.path.join(OUTPUT_FOLDER_NAME, "Data_to_label")
    dst_dir = os.path.join(OUTPUT_FOLDER_NAME, "Filtered")
    os.makedirs(batch_dir, exist_ok=True)
    os.makedirs(dst_dir)

    temp_tsv = filter_and_sort_tsv()
    index = 0

    EOF = get_next_batch(temp_tsv, index)
    index += BATCH_SIZE

    while True:
        user_input = input("Press 'enter' to save current batch, press 'q' to quit \n")
        if user_input == "q":
            break

        if user_input == "":
            approved_files = [f for f in os.listdir(batch_dir) if f.endswith(".mp3")]
            for f in approved_files:
                shutil.move(
                    os.path.join(batch_dir, f),
                    os.path.join(dst_dir, f),
                )

            with open(temp_tsv, "r", encoding="utf-8") as f_in, open(
                os.path.join(OUTPUT_FOLDER_NAME, "labeled.tsv"), "a", encoding="utf-8"
            ) as f_out:
                for line in f_in:
                    if line.split("\t")[0] in approved_files:
                        f_out.write(line)

            if count_files(dst_dir) >= SIZE:
                break

            EOF = get_next_batch(temp_tsv, index)
            index += BATCH_SIZE

        if EOF:
            break

    shutil.rmtree(batch_dir)


import argparse

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--size",
        type=int,
        default=200,
        help="Size of your desired dataset",
    )
    parser.add_argument(
        "--accent",
        type=str,
        help="Choose accent to label",
    )

    args = parser.parse_args()
    SIZE = args.size

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

    if args.accent:
        ACCENT_LABEL = args.accent
    else:
        print("Choose an accent to label:")
        for i, accent in enumerate(ACCENTS, 1):
            print(f"{i}. {accent}")
        while True:
            try:
                choice = int(input("Enter number of desired accent: "))
                if 1 <= choice <= len(ACCENTS):
                    ACCENT_LABEL = ACCENTS[choice - 1]
                    break
            except ValueError:
                pass
            print("Invalid choice. Please enter a number from the list.")

    OUTPUT_FOLDER_NAME = ACCENT_LABEL.strip().split()[0] + "_dataset"

    main()
