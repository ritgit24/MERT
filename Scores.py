import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score


def get_label_name_from_filename(filename):
    filename = filename.lower().strip()

    if ".mp3_chunk_" in filename:
        return filename.split(".mp3_chunk_")[0]

    if "_chunk_" in filename:
        return filename.split("_chunk_")[0]

    return filename.replace(".wav", "")


def evaluate_embedding_space(npy_file, name):
    data = np.load(npy_file, allow_pickle=True).item()

    filenames = list(data.keys())
    embeddings = np.array(list(data.values()))

    labels = [
        get_label_name_from_filename(f)
        for f in filenames
    ]

    unique_labels = sorted(set(labels))
    label_to_id = {
        label: idx
        for idx, label in enumerate(unique_labels)
    }

    y = np.array([label_to_id[label] for label in labels])

    sil = silhouette_score(embeddings, y)
    db = davies_bouldin_score(embeddings, y)

    print(f"\n{name}")
    print("-" * 40)
    print(f"Silhouette Score     : {sil:.4f}")
    print(f"Davies-Bouldin Score : {db:.4f}")

    return sil, db


if __name__ == "__main__":

    evaluate_embedding_space(
        "base_mert_developed_embeddings.npy",
        "Base MERT"
    )

    evaluate_embedding_space(
        "cultural_mert_developed_embeddings.npy",
        "Cultural MERT"
    )