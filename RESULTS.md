# Results and Conclusion
The original MERT repository was used as the base implementation. The core MERT architecture and pretrained layers were retained, so the model continued to use the representation-learning capability of the original MERT backbone.

The main changes were made around the dataset and training pipeline. Instead of using the original data setup directly, I modified the pipeline to work with the Saraga raga classification dataset. Audio recordings were split into fixed-length chunks, labels were extracted from the chunk filenames, and the data loader was adapted to support supervised raga classification across all available ragas.

## Classification Performance

The Cultural MERT model was evaluated on a multi-class raga classification task consisting of **83 distinct ragas**.

### Performance Metrics

| Metric         | Score     |
| -------------- | --------- |
| Top-1 Accuracy | **77.2%** |
| Top-5 Accuracy | **91.5%** |

The large gap between Top-1 and Top-5 accuracy suggests that even when the model's first prediction is incorrect, it frequently ranks the correct raga among its most likely candidates.

---

## Embedding Space Analysis

To evaluate the quality of the learned representations, embeddings were extracted from both the pretrained Base MERT model and the fine-tuned Cultural MERT model.

Each audio chunk was represented by a **768-dimensional embedding vector**.

### Evaluation on All 83 Ragas

| Model         | Silhouette Score | Davies–Bouldin Score |
| ------------- | ---------------- | -------------------- |
| Base MERT     | -0.0426          | 2.3048               |
| Cultural MERT | 0.0015           | 2.2397               |

The Cultural MERT embeddings showed a slight improvement over the pretrained baseline.

---

## Filtered Evaluation

Many ragas contained very few samples, which can introduce noise into clustering-based metrics. To obtain a more reliable assessment, ragas with fewer than **5 samples** were removed.

### Filtered Dataset Statistics

| Statistic                | Value |
| ------------------------ | ----- |
| Minimum Samples per Raga | 5     |
| Remaining Ragas          | 40    |
| Remaining Embeddings     | 335   |

### Filtered Embedding Metrics

| Model         | Silhouette Score | Davies–Bouldin Score |
| ------------- | ---------------- | -------------------- |
| Base MERT     | 0.0104           | 2.5130               |
| Cultural MERT | 0.0426           | 2.4349               |

After removing low-support classes, the improvement became more evident. Cultural MERT achieved a higher silhouette score and a lower Davies–Bouldin score, indicating better cluster compactness and separation.

---

## Conclusion

The results demonstrate that Cultural MERT successfully learned more domain-specific representations than the original pretrained model. Evidence for this improvement is observed across both downstream classification performance and embedding-space quality.

Key findings include:

* **77.2% Top-1 Accuracy** across 83 ragas.
* **91.5% Top-5 Accuracy** on the test set.
* Improved clustering quality compared to Base MERT.
* Clearer gains after filtering extremely low-support ragas.
* More compact same-raga clusters and improved separation between different ragas.

Overall, the experiments suggest that fine-tuning MERT on culturally specific music data enables the model to capture meaningful raga-related characteristics and produce more informative musical representations for downstream tasks.
