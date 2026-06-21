import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def get_label_name_from_filename(filename):
    filename = filename.lower().strip()

    # Match the label string extraction mechanism from your training pipeline
    if ".mp3_chunk_" in filename:
        return filename.split(".mp3_chunk_")[0]

    if "_chunk_" in filename:
        return filename.split("_chunk_")[0]

    return filename.replace(".wav", "")


def plot_latent_space(npy_path, title):
    print(f"Reading embedding matrix data from target path: {npy_path}")
    data = np.load(npy_path, allow_pickle=True).item()

    filenames = list(data.keys())
    vectors = np.array(list(data.values()))

    print(f"Successfully loaded {len(vectors)} multidimensional structural embeddings.")

    labels = [
        get_label_name_from_filename(fname)
        for fname in filenames
    ]

    # Dynamically optimize perplexity relative to your test dataset sample allocation density
    perplexity = min(30, max(5, len(vectors) // 3))
    print(f"Computing t-SNE projections using optimized perplexity coefficient: {perplexity}")

    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=perplexity,
        max_iter=1000  # Fixed: Changed from n_iter to max_iter
    )

    reduced_vectors = tsne.fit_transform(vectors)

    plt.figure(figsize=(14, 10))

    unique_labels = sorted(set(labels))
    print(f"Mapping {len(unique_labels)} unique Raga class labels into visual space color nodes...")

    for label in unique_labels:
        idx = [i for i, l in enumerate(labels) if l == label]

        plt.scatter(
            reduced_vectors[idx, 0],
            reduced_vectors[idx, 1],
            label=label,
            alpha=0.8,   # Slightly higher opacity to clearly isolate class groupings
            s=35         # Slightly larger point size for clean visualization
        )

    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("t-SNE Dimension 1", fontsize=11)
    plt.ylabel("t-SNE Dimension 2", fontsize=11)

    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8,
        title="Raga Classes",
        title_fontsize=9,
        ncol=2 if len(unique_labels) > 15 else 1  # Spreads out legend if you have many classes
    )

    plt.grid(True, linestyle='--', alpha=0.5)

    output_file = (
        title.lower()
        .replace(" ", "_")
        .replace("/", "_")
        + ".png"
    )

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\n[Success] Visualization saved perfectly as: {output_file}")


if __name__ == "__main__":
    # Configured to look for the exact .npy array written by your gradual training loop run
    plot_latent_space(
        "./cultural_mert_gradual_developed_embeddings.npy",
        "Cultural MERT Gradual Unfreezing Embedding Space"
    )
