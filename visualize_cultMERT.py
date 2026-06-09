import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def get_label_name_from_filename(filename):
    filename = filename.lower().strip()

    if ".mp3_chunk_" in filename:
        return filename.split(".mp3_chunk_")[0]

    if "_chunk_" in filename:
        return filename.split("_chunk_")[0]

    return filename.replace(".wav", "")


def plot_latent_space(npy_path, title):
    data = np.load(npy_path, allow_pickle=True).item()

    filenames = list(data.keys())
    vectors = np.array(list(data.values()))

    print(f"Loaded {len(vectors)} embeddings")

    labels = [
        get_label_name_from_filename(fname)
        for fname in filenames
    ]

    perplexity = min(30, max(5, len(vectors) // 3))

    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=perplexity
    )

    reduced_vectors = tsne.fit_transform(vectors)

    plt.figure(figsize=(14, 10))

    unique_labels = sorted(set(labels))

    for label in unique_labels:
        idx = [i for i, l in enumerate(labels) if l == label]

        plt.scatter(
            reduced_vectors[idx, 0],
            reduced_vectors[idx, 1],
            label=label,
            alpha=0.7,
            s=25
        )

    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")

    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=7
    )

    plt.grid(True)

    output_file = (
        title.lower()
        .replace(" ", "_")
        .replace("/", "_")
        + ".png"
    )

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    plot_latent_space(
        "./cultural_mert_developed_embeddings.npy",
        "Cultural MERT Embedding Space"
    )