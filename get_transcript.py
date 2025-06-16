from pathlib import Path

list_path = Path("list.txt")
tsv_path = Path("validated.tsv")
output_path = Path("transcripts.txt")

with open(list_path) as f:
    filenames = set(line.strip() for line in f)

with open(tsv_path) as tsv, open(output_path, "w") as out:
    next(tsv)  # skip header
    for line in tsv:
        parts = line.strip().split("\t")
        if len(parts) >= 5 and parts[1] in filenames:
            out.write(parts[1] + "\t" + parts[3] + "\n")

