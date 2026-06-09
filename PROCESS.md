## Preprocessing 
The hindustani audio files have been used for the following computation. 

After preprocessing and chunking the audio recordings into fixed-length segments, the final classification task consisted of 83 distinct ragas.
Some ragas had many examples while others had very few.
Several ragas are musically similar and difficult to distinguish, even for human listeners.

Each audio file was segmented into fixed-length chunks.Rather than restricting the task to a small subset of ragas, all 83 available ragas were retained , creating a challenging multi-class classification problem with significant class imbalance and high inter-raga similarity. This setup provided a more realistic benchmark for evaluating whether MERT could learn culturally meaningful representations of Indian classical music.

The chunk filenames were designed to preserve the original raga information so that labels could later be recovered automatically. The dataset pipeline then extracted labels from the chunk filenames.
Generated label-to-index mappings for classification.

This allowed the model to process all ragas using a unified training pipeline.

## Extracting Base MERT Embeddings

Before performing any fine-tuning, the pretrained MERT model was used to generate embeddings for the dataset.

For each audio chunk:

The chunk was passed through MERT.
The hidden representations were extracted.
A fixed-length embedding vector was generated and stored(with a dimension of 768 x 1)

These embeddings served as a baseline representation of the music before any cultural adaptation.

Then MERT was fine-tuned on the raga classification task.

The objective was to adapt the pretrained music representation model to better capture characteristics specific to Indian classical music.

During training:

Audio chunks were passed through MERT.
A classification head predicted the corresponding raga.
Cross-entropy loss was optimized.
The model weights were updated through backpropagation.

Training logs showed gradual learning, although the loss remained somewhat noisy due to the small batch size and large number of classes.

Total 10 epochs were run.

## Evaluating Classification Performance

After training ans testing,

The model achieved:

Top-1 Accuracy: 77.2%
Top-5 Accuracy: 91.5%

The gap between 77.2% Top-1 Accuracy and 91.5% Top-5 Accuracy suggests that the model usually narrows its predictions down to a small set of musically similar ragas, even when its first choice is incorrect. After this the embeddings were generated for Cultural MERT model.

**Visualizing the Embedding Space** To qualitatively compare both models, t-SNE dimensionality reduction was applied to the embeddings.

The resulting plots provided a two-dimensional visualization of the embedding spaces.

Although some local grouping could be observed, the large number of ragas (83 classes) made visual interpretation difficult. The t-SNE plots alone were insufficient to conclusively determine whether the Cultural MERT embeddings were better.

## Quantitative Embedding Evaluation

The two metrics used were: **Silhouette Score**  and **Davies–Bouldin Score**

Using all 83 ragas, the results were:

Base MERT:

Silhouette Score: -0.0426
Davies–Bouldin Score: 2.3048

Cultural MERT:

Silhouette Score: 0.0015
Davies–Bouldin Score: 2.2397

These results suggested a slight improvement in the Cultural MERT embedding space.

However, the gains were relatively small.

## Investigating Class Imbalance

It was observed that many ragas had extremely small support values.

Several classes contained only:

1 sample
2 samples
3 samples
4 samples

Such small classes can introduce significant noise into clustering metrics and make evaluation unreliable.

Additionally, some ragas are musically very similar, making separation difficult even for expert listeners when only a short audio segment is available.

## Filtering Low-Support Ragas

To obtain a cleaner evaluation, the embedding analysis was repeatedwhile retaining only ragas with at least 5 samples.

This reduced the dataset to: 40 ragas, 335 embeddings and removed many of the noisiest classes.

## Recomputing Embedding Metrics

After filtering, the results became:

Base MERT:

Silhouette Score: 0.0104
Davies–Bouldin Score: 2.5130

Cultural MERT:

Silhouette Score: 0.0426
Davies–Bouldin Score: 2.4349

The silhouette score increased by roughly four times, while the Davies–Bouldin score decreased, indicating that Cultural MERT produced more compact same-raga clusters and better separation between different ragas.

## Final Conclusion

The Cultural MERT model successfully adapted the pretrained MERT representations to the Indian classical music domain.

Evidence for this improvement came from multiple sources:

77.2% classification accuracy.
91.5% top-5 accuracy.
Improved silhouette score.
Improved Davies–Bouldin score.
Better clustering behavior after removing extremely low-support ragas.

Overall, the results suggest that fine-tuning MERT on culturally specific music data helps the model learn more meaningful representations of ragas and improves downstream classification performance.