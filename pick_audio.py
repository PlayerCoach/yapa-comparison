import argparse
import os
import re
import shutil

BIG_DATA_PATH = "data/cv-corpus-21.0-2025-03-14/en"
SMALL_DATA_PATH = "data/cv-corpus-20.0-delta-2024-12-06/en"

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


def create_set(data_path, accent_regex, name, size, output_dir, inject_polish=False):
    tsv_path = os.path.join(data_path, TSV_FILE)
    clips_path = os.path.join(data_path, CLIPS_DIR)
    output_dir = os.path.join(output_dir, f"{name}_clips")
    output_tsv = os.path.join(output_dir, f"{name}.tsv")

    os.makedirs(output_dir, exist_ok=True)

    regex = re.compile(accent_regex, re.IGNORECASE)
    polish_regex = re.compile(r"pol", re.IGNORECASE)

    entries = []
    polish_entries = []

    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = parse_line(line)
            if not entry:
                continue

            if inject_polish and polish_regex.search(entry["accent"]):
                polish_entries.append(entry)
            elif regex.search(entry["accent"]):
                score = entry["upvotes"] - entry["downvotes"]
                entries.append((score, entry))

    entries.sort(key=lambda x: (x[0] != 0, -x[0]))
    sorted_entries = [e for _, e in entries]

    if inject_polish:
        full_list = polish_entries + sorted_entries
    else:
        full_list = sorted_entries

    used_clients = set()
    selected = []

    for entry in full_list:
        if entry["client_id"] in used_clients:
            continue
        # check if audio is of current length
        if not os.path.exists(os.path.join(clips_path, entry["filename"])):
            print(f"⚠️ File not found: {entry['filename']}")
            continue
        # check if audio is of current length
        import soundfile as sf

        info = sf.info(clips_path, entry["filename"])
        min_samples = 512
        if info.frames < min_samples:
            continue
        used_clients.add(entry["client_id"])
        selected.append(entry)
        if len(selected) >= size:
            break

    with open(output_tsv, "w", encoding="utf-8") as out:
        for entry in selected:
            out.write(
                f"{entry['filename']}\t{entry['transcript']}\t{entry["upvotes"] - entry["downvotes"]}\t{entry['accent']}\n"
            )
            src = os.path.join(clips_path, entry["filename"])
            dst = os.path.join(output_dir, entry["filename"])
            if os.path.exists(src):
                shutil.copy(src, dst)
            else:
                print(f"⚠️ File not found: {src}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--size",
        type=int,
        default=10,
        help="Size of each group British | American | Other, dataset is of size 3*size",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/dataset",
        help="Output directory path for the dataset",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="big_data",
        help="Chooses from the smaller or bigger dataset",
    )

    args = parser.parse_args()

    if args.input == "big_data":
        DATA_PATH = BIG_DATA_PATH
    elif args.input == "small_data":
        DATA_PATH = SMALL_DATA_PATH
    else:
        raise ValueError(
            "Invalid input dataset choice. Use 'big_data' or 'small_data'."
        )

    print(f"Using dataset: {DATA_PATH}")
    print(f"Output directory: {args.output}")
    print(f"Size of each group: {args.size}")

    create_set(DATA_PATH, r"^England English$", "british", args.size, args.output)
    create_set(
        DATA_PATH, r"^United States English$", "american", args.size, args.output
    )
    create_set(
        DATA_PATH,
        r"^(?!.*(england|united states|british)).*$",
        "other",
        args.size,
        args.output,
        inject_polish=True,
    )


def pick_audio(size=1000, output="data/new_dataset", input="big_data"):
    if input == "big_data":
        DATA_PATH = BIG_DATA_PATH
    elif input == "small_data":
        DATA_PATH = SMALL_DATA_PATH
    else:
        raise ValueError(
            "Invalid input dataset choice. Use 'big_data' or 'small_data'."
        )
    print(f"Using dataset: {DATA_PATH}")
    print(f"Output directory: {output}")
    print(f"Size of each group: {size}")
    print("Creating dataset...")
    create_set(DATA_PATH, r"^England English$", "british", size, output)
    create_set(DATA_PATH, r"^United States English$", "american", size, output)
    # create_set(DATA_PATH, r"^Irish English$", "irish", size, output)
    # create_set(DATA_PATH, r"^Scottish English$", "scottish", size, output)
    # create_set(DATA_PATH, r"^Australian English$", "australian", size, output)
    # create_set(DATA_PATH, r"^Filipino$", "filipino", size, output)
    # create_set(
    #     DATA_PATH,
    #     r"^India and South Asia \(India, Pakistan, Sri Lanka\)$",
    #     "indian",
    #     size,
    #     output,
    # )
    # create_set(
    #     DATA_PATH,
    #     r"^Canadian English$",
    #     "canadian",
    #     size,
    #     output,
    # )

    # create_set(
    #     DATA_PATH,
    #     r"^(?!.*(england|united states|british)).*$",
    #     "other",
    #     size,
    #     output,
    #     inject_polish=True,
    # )
    print("Dataset created successfully.")
