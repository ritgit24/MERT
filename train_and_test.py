import os
import torch
import soundfile as sf
import librosa
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, AutoModel
from sklearn.metrics import accuracy_score, classification_report, top_k_accuracy_score

BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4
EPOCHS = 10
MAX_AUDIO_SECONDS = 5
SAMPLE_RATE = 24000
MAX_SAMPLES = MAX_AUDIO_SECONDS * SAMPLE_RATE


def get_label_name_from_filename(filename):
    filename = filename.lower().strip()

    if ".mp3_chunk_" in filename:
        return filename.split(".mp3_chunk_")[0]

    if "_chunk_" in filename:
        return filename.split("_chunk_")[0]

    return filename.replace(".wav", "")


class SaragaDataset(Dataset):
    def __init__(self, tsv_path, processor, label_map=None):
        self.processor = processor
        self.audio_entries = []

        with open(tsv_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
            self.base_dir = lines[0]

            for line in lines[1:]:
                if "\t" in line:
                    rel_path, _ = line.split("\t")
                    self.audio_entries.append(os.path.join(self.base_dir, rel_path))

        if label_map is None:
            self.label_map = {}

            for path in self.audio_entries:
                label_name = get_label_name_from_filename(os.path.basename(path))

                if label_name not in self.label_map:
                    self.label_map[label_name] = len(self.label_map)
        else:
            self.label_map = label_map

        self.num_classes = len(self.label_map)

    def __len__(self):
        return len(self.audio_entries)

    def __getitem__(self, idx):
        file_path = self.audio_entries[idx]

        audio, sr = sf.read(file_path)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if sr != SAMPLE_RATE:
            audio = librosa.resample(
                audio,
                orig_sr=sr,
                target_sr=SAMPLE_RATE
            )

        audio = audio[:MAX_SAMPLES]
        waveform = torch.tensor(audio, dtype=torch.float32)

        label_name = get_label_name_from_filename(os.path.basename(file_path))
        label = self.label_map[label_name]

        return waveform, label


class MERTClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype=torch.float16,
        )

        for param in self.backbone.parameters():
            param.requires_grad = False

        hidden = self.backbone.config.hidden_size
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = x.to(dtype=torch.float16)

        dev_type = "cuda" if torch.cuda.is_available() else "cpu"

        with torch.autocast(device_type=dev_type):
            outputs = self.backbone(
                input_values=x,
                output_hidden_states=True
            )

        embeddings = torch.mean(outputs.last_hidden_state.float(), dim=1)
        logits = self.classifier(embeddings)

        return logits, embeddings


def run_pipeline(model_name, run_label):
    print(f"\n STARTING RUN: {run_label} ")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Active Processing Hardware Target: {device.upper()}")

    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    def collate_fn(batch):
        waveforms = [wave.numpy() for wave, label in batch]
        labels = [label for wave, label in batch]

        inputs = processor(
            waveforms,
            sampling_rate=SAMPLE_RATE,
            padding=True,
            return_tensors="pt"
        )

        return inputs.input_values, torch.tensor(labels, dtype=torch.long)

    temp_train_ds = SaragaDataset("./saraga_data/manifests/train.tsv", processor)
    temp_test_ds = SaragaDataset("./saraga_data/manifests/test.tsv", processor)

    combined_label_map = {}

    for path in temp_train_ds.audio_entries + temp_test_ds.audio_entries:
        label_name = get_label_name_from_filename(os.path.basename(path))

        if label_name not in combined_label_map:
            combined_label_map[label_name] = len(combined_label_map)

    train_ds = SaragaDataset(
        "./saraga_data/manifests/train.tsv",
        processor,
        label_map=combined_label_map
    )

    test_ds = SaragaDataset(
        "./saraga_data/manifests/test.tsv",
        processor,
        label_map=combined_label_map
    )

    print("\nDetected classes:")
    for name, idx in train_ds.label_map.items():
        print(idx, "->", name)

    print(f"\nTrain chunks: {len(train_ds)}")
    print(f"Test chunks : {len(test_ds)}")
    print(f"Classes     : {train_ds.num_classes}")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    model = MERTClassifier(
        model_name,
        num_classes=train_ds.num_classes
    ).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=2e-4
    )

    criterion = nn.CrossEntropyLoss()

    print("\nTraining downstream classification layer...")
    model.train()
    optimizer.zero_grad()

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        for step, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits, _ = model(inputs)

            loss = criterion(logits, labels) / GRAD_ACCUM_STEPS
            loss.backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

            if step % 20 == 0:
                print(
                    f"  Step {step:4d} | Current Step Loss: "
                    f"{loss.item() * GRAD_ACCUM_STEPS:.4f}"
                )

            del inputs, labels, logits

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\nEvaluating model variants and saving developed embeddings...")

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    extracted_embeddings = {}

    with torch.no_grad():
        for step, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)

            logits, embeddings = model(inputs)

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

            for idx, emb in enumerate(embeddings.cpu().numpy()):
                global_idx = step * BATCH_SIZE + idx

                if global_idx < len(test_ds.audio_entries):
                    fname = os.path.basename(test_ds.audio_entries[global_idx])
                    extracted_embeddings[fname] = emb

            del inputs, logits, embeddings

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    np.save(f"./{run_label}_developed_embeddings.npy", extracted_embeddings)

    all_probs = np.array(all_probs)

    print(f"\n--- Evaluation Metrics Result Logs for: {run_label} ---")
    print(f"Accuracy Score: {accuracy_score(all_labels, all_preds):.4f}")

    if train_ds.num_classes >= 5:
        print(
            f"Top-5 Accuracy: "
            f"{top_k_accuracy_score(all_labels, all_probs, k=5, labels=list(range(train_ds.num_classes))):.4f}"
        )

    print(classification_report(all_labels, all_preds, zero_division=0))


if __name__ == "__main__":
    run_pipeline("m-a-p/MERT-v1-95M", "base_mert")