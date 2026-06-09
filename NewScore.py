import numpy as np
from collections import Counter
from sklearn.metrics import silhouette_score, davies_bouldin_score


def get_label_name_from_filename(filename):
    filename = filename.lower().strip()

    if ".mp3_chunk_" in filename:
        return filename.split(".mp3_chunk_")[0]

    if "_chunk_" in filename:
        return filename.split("_chunk_")[0]

    return filename.replace(".wav", "")


def evaluate_filtered_embedding_space(
    npy_file,
    name,
    min_samples=5
):
    data = np.load(npy_file, allow_pickle=True).item()

    filenames = list(data.keys())
    embeddings = np.array(list(data.values()))

    labels = [
        get_label_name_from_filename(f)
        for f in filenames
    ]

    label_counts = Counter(labels)

    valid_labels = {
        label
        for label, count in label_counts.items()
        if count >= min_samples
    }

    filtered_embeddings = []
    filtered_labels = []

    for emb, label in zip(embeddings, labels):
        if label in valid_labels:
            filtered_embeddings.append(emb)
            filtered_labels.append(label)

    filtered_embeddings = np.array(filtered_embeddings)

    unique_labels = sorted(set(filtered_labels))

    label_to_id = {
        label: idx
        for idx, label in enumerate(unique_labels)
    }

    y = np.array([
        label_to_id[label]
        for label in filtered_labels
    ])

    sil = silhouette_score(
        filtered_embeddings,
        y
    )

    db = davies_bouldin_score(
        filtered_embeddings,
        y
    )

    print(f"\n{name}")
    print("-" * 50)
    print(f"Minimum samples per raga : {min_samples}")
    print(f"Remaining ragas          : {len(unique_labels)}")
    print(f"Remaining embeddings     : {len(filtered_embeddings)}")
    print(f"Silhouette Score         : {sil:.4f}")
    print(f"Davies-Bouldin Score     : {db:.4f}")

    return sil, db


if __name__ == "__main__":

    evaluate_filtered_embedding_space(
        "base_mert_developed_embeddings.npy",
        "Base MERT",
        min_samples=5
    )

    evaluate_filtered_embedding_space(
        "cultural_mert_developed_embeddings.npy",
        "Cultural MERT",
        min_samples=5
    )