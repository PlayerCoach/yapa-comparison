import os
import re
import shutil

TSV_FILE = "validated.tsv"
OUTPUT_DIR = "other_accents"
CLIPS_DIR = "clips"
OUTPUT_LIST = "other_accents_subset.txt"
MAX_FILES = 712

EXCLUDE_PATTERN = re.compile(r"(united states|british|england|pol)", re.IGNORECASE)

entries_by_client = {}

with open(TSV_FILE, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 10:
            client_id = parts[0]
            filename = parts[1]
            transcript = parts[3]
            accent = parts[9]
            if not EXCLUDE_PATTERN.search(accent):
                if client_id not in entries_by_client:
                    entries_by_client[client_id] = (filename, transcript)
                if len(entries_by_client) >= MAX_FILES:
                    break

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(OUTPUT_LIST, "w", encoding="utf-8") as out:
    for filename, transcript in entries_by_client.values():
        out.write(f"{filename}\t{transcript}\n")
        src = os.path.join(CLIPS_DIR, filename)
        dst = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            print(f"⚠️ File not found: {src}")

