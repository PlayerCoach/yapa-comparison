import shutil
from pathlib import Path

base_path = Path.cwd() / "data/cv-corpus-21.0-2025-03-14/en/clips"
destination = Path.cwd() / "polish_clips"

destination.mkdir(exist_ok=True)

with open("list.txt") as f:
    for line in f:
        filename = line.strip()
        src = base_path / filename
        dst = destination / filename
        if src.exists():
            print(f"Copying {src} to {dst}")
            shutil.copy(src, dst)
        else:
            print(f"File not found: {src}")

