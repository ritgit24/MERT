import os
import glob
import random
from collections import defaultdict
import soundfile as sf


def get_label_name_from_filename(filename):
    filename = filename.lower().strip()

    if ".mp3_chunk_" in filename:
        return filename.split(".mp3_chunk_")[0]

    if "_chunk_" in filename:
        return filename.split("_chunk_")[0]

    return filename.replace(".wav", "")


def generate_mert_manifests(chunks_dir, output_dir, train_ratio=0.8, valid_ratio=0.1):
    os.makedirs(output_dir, exist_ok=True)

    all_chunks = glob.glob(os.path.join(chunks_dir, "*.wav"))

    if not all_chunks:
        print(f"Error: No .wav chunk files found in {chunks_dir}!")
        return

    print(f"Found {len(all_chunks)} total chunks on disk. Grouping by raga label...")

    label_groups = defaultdict(list)

    for chunk_path in all_chunks:
        filename = os.path.basename(chunk_path)
        label_name = get_label_name_from_filename(filename)
        label_groups[label_name].append(chunk_path)

    print(f"Identified {len(label_groups)} unique labels.")

    splits = {
        "train": [],
        "valid": [],
        "test": []
    }

    random.seed(42)

    for label_name, files in label_groups.items():
        random.shuffle(files)

        n = len(files)

        train_end = int(n * train_ratio)
        valid_end = train_end + int(n * valid_ratio)

        # For tiny classes, keep at least one train sample
        if train_end == 0 and n > 0:
            train_end = 1

        train_files = files[:train_end]
        valid_files = files[train_end:valid_end]
        test_files = files[valid_end:]

        splits["train"].extend(train_files)
        splits["valid"].extend(valid_files)
        splits["test"].extend(test_files)

    print("\n--- Data Split Allocation Results ---")
    print(f"Train set: {len(splits['train'])} chunks")
    print(f"Valid set: {len(splits['valid'])} chunks")
    print(f"Test set : {len(splits['test'])} chunks")

    abs_base_dir = os.path.abspath(chunks_dir)

    for split_name, chunk_list in splits.items():
        output_tsv_path = os.path.join(output_dir, f"{split_name}.tsv")

        with open(output_tsv_path, "w", encoding="utf-8") as tsv_file:
            tsv_file.write(f"{abs_base_dir}\n")

            successful_writes = 0

            for filepath in chunk_list:
                rel_name = os.path.basename(filepath)

                try:
                    info = sf.info(filepath)
                    total_frames = info.frames

                    if total_frames <= 0:
                        continue

                    tsv_file.write(f"{rel_name}\t{total_frames}\n")
                    successful_writes += 1

                except Exception:
                    continue

        print(
            f"Manifest saved successfully: {output_tsv_path} "
            f"({successful_writes} entries written)"
        )


CHUNKS_DIR_PATH = "./saraga_data/processed_chunks"
MANIFEST_OUTPUT_DIR = "./saraga_data/manifests"


if __name__ == "__main__":
    generate_mert_manifests(CHUNKS_DIR_PATH, MANIFEST_OUTPUT_DIR)