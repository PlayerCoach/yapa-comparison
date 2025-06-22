import argparse
import os
import re
import shutil

BIG_DATA_PATH = "../data/cv-corpus-21.0-2025-03-14/en"
SMALL_DATA_PATH = "../data/cv-corpus-20.0-delta-2024-12-06/en"
TEST_DATA_PATH = "../data/cv-corpus-10.0-delta-2022-07-04/en"

TSV_FILE = "validated.tsv"
CLIPS_DIR = "clips"


def parse_line(line):
    parts = line.strip().split("\t")
    if len(parts) < 10 or parts[0] == "client_id":
        return None
    return {
        "client_id": parts[0],
        "filename": parts[1],
        "transcript": parts[3],
        "upvotes": int(parts[5]) if parts[5].isdigit() else 0,
        "downvotes": int(parts[6]) if parts[6].isdigit() else 0,
        "accent": parts[9],
    }


def count_unique_speakers():
    ACCENTS = [
        "Australian English",
        "Canadian English",
        "England English",
        "India and South Asia (India, Pakistan, Sri Lanka)",
        "Irish English",
        "Scottish English",
        "United States English",
        "Filipino",
        "Slavic",  # This is for my special use, you can delete it if you want
        # Add more here
    ]

    tsv_path = os.path.join(TEST_DATA_PATH, TSV_FILE)

    for accent in ACCENTS:
        # If you don't want to count slavic counteries you can delete this if/else
        if accent == "Slavic":
            regex = re.compile(
                r"\b(Slavic|Polish|Czech|Russian|Ukrainian|Bulgarian| \
                               Croatian|Slovak|Slovenian|Serbian|Latvian|Lithuanian| \
                               Hungarian|Romanian|Kazakh|Azerbaijani|Georgian|Moldovan)\b",
                re.IGNORECASE,
            )
        else:
            regex = re.compile(rf"^{re.escape(accent)}$")

        seen_speakers = set()
        counter = 0

        with open(tsv_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = parse_line(line)
                if entry is None:
                    continue
                if regex.search(entry["accent"]):
                    if entry["client_id"] not in seen_speakers:
                        counter += 1
                        seen_speakers.add(entry["client_id"])

        print(f"Uniqe speakers for {accent} : {counter}")


if __name__ == "__main__":
    count_unique_speakers()
