# Project Results: Adapting CultureMERT for Indian Classical Raga Classification

This document summarizes the performance, challenges, and current limitations of fine-tuning the **CultureMERT (95M)** music transformer model on Indian Classical Ragas using a **4-layer neural network head with gradual unfreezing**.

---

##  Performance Metrics

The model achieved an outstanding performance leap after fixing optimization bottlenecks:

*   **Overall Classification Accuracy**: **75.87%**
*   **Top-5 Accuracy**: **93.26%** 
    *   *Meaning*: Even if the model misses the exact Raga on its first guess, the correct Raga is almost always within its top 5 choices.
*   **High-Performing Classes**: Several Ragas achieved a perfect **1.00 Precision and Recall**, meaning the model completely mastered their unique microtonal transitions (*swaras*).

---

## The exploding gradient problem 

During the initial training runs, the model crashed at the start of **Epoch 3**—the exact moment the first pre-trained layers of the MERT model were unfrozen. The loss values immediately turned into **`nan` (Not a Number)**.

1. **Optimizer Memory Wipe**: In early code iterations, every time the model hit an unfreezing milestone, the code threw away the old optimizer and created a new one from scratch. This completely erased the optimizer's active momentum memory. The model essentially "forgot" how to learn every two epochs.
2. **Batch Size Noise**: Due to Laptop GPU VRAM limits (6GB), we were forced to train with a tiny batch size (`BATCH_SIZE = 2`). The network was using `Batch Normalization`, which calculates math averages across a batch. With only 2 samples, the math averages were wildly inaccurate and unstable, causing deep structural noise.
3. **Floating Point Overflow**: The MERT model was loaded in 16-bit float precision (`FP16`). The massive feedback spike from the broken optimization path pushed the numbers past what `FP16` can handle, rolling them over into infinity (`nan`).

To solve this : 

*   **Switched to Layer Normalization**: Replaced `BatchNorm1d` with `LayerNorm`. Layer Normalization calculates math internally per audio file instead of across batches, making it immune to tiny batch sizes.
*   **Preserved Optimizer Momentum**: Rewrote the scheduler to use `optimizer.add_param_group()`. Instead of destroying the optimizer, it gently appends newly unfrozen layers while keeping the momentum of the rest of the model intact.
*   **Gradient Clipping & Lower Learning Rates**: We clamped the maximum size of gradient steps to `1.0` and dropped the backbone learning rate down to an ultra-safe speed (`1e-6`) to protect pre-trained weights.

---

*Epoch wise Progress : *

| Epoch | Backbone Status | Train Loss | Valid Loss | Valid Acc |
| :---: | :--- | :---: | :---: | :---: |
| **1** | Fully Frozen | 4.0603 | 3.4605 | 30.09% |
| **2** | Fully Frozen | 3.1683 | 2.6368 | 43.27% |
| **3** | Unfreezing Layers 9–11 | 2.3992 | 1.9975 | 57.88% |
| **4** | Unfreezing Layers 9–11 | 1.8724 | 1.5699 | 64.76% |
| **5** | Unfreezing Layers 6–8 | 1.4591 | 1.2458 | 71.06% |
| **6** | Unfreezing Layers 6–8 | 1.1906 | 0.9984 | 74.21% |
| **7** | Unfreezing Layers 3–5 | 0.9947 | 0.9056 | 75.93% |
| **8** | Unfreezing Layers 3–5 | 0.8194 | 0.7655 | 78.22% |
| **9** | Entire Backbone Open | 0.7259 | 0.7656 | 77.94% |
| **10** | Entire Backbone Open | 0.6312 | 0.6841 | 77.94% |


##  Where is it Lacking Right Now

Though we have 75.8% accuracy, the t-SNE visualization reveals clear areas where the model is still struggling:

### 1. The Dataset Contains Non-Musical Data
The biggest limitation is that the dataset contains speech alongside music. The labels include things like *"introduction about ajoy chakrabarty"* and *"introductory speech"*. 
*   **The Issue**: MERT is trained exclusively on music. When it is forced to process spoken-word lectures, it gets confused. This is why classes like Class 0, 19, and 21 scored a **0% accuracy**. They aren't actually music.

### 3. Data Starvation for Rare Classes
Ragas with plenty of audio samples perform flawlessly, but classes with only 2 or 3 test samples collapse. The model ignores them during training to focus on classes where it can win more global accuracy points.

---