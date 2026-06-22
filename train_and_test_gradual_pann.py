import os
import torch
import soundfile as sf
import librosa
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, AutoModel
from sklearn.metrics import accuracy_score, classification_report, top_k_accuracy_score

# Try loading PANNs package dynamically
try:
    from panns_inference import AudioTagging
    PANN_AVAILABLE = True
except ImportError:
    PANN_AVAILABLE = False


# ──────────────────────────────────────────────
# Hyperparameters & Global Config
# ──────────────────────────────────────────────
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4
EPOCHS = 10
MAX_AUDIO_SECONDS = 5
SAMPLE_RATE = 24000
MAX_SAMPLES = MAX_AUDIO_SECONDS * SAMPLE_RATE

# PANN models expect audio at a strict 32,000Hz baseline
PANN_SAMPLE_RATE = 32000
SPEECH_THRESHOLD = 0.30  # Discard if speech probability > 30%


# ──────────────────────────────────────────────
# Label Extraction
# ──────────────────────────────────────────────
def get_label_name_from_filename(filename):
    filename = filename.lower().strip()
    if ".mp3_chunk_" in filename:
        return filename.split(".mp3_chunk_")[0]
    if "_chunk_" in filename:
        return filename.split("_chunk_")[0]
    return filename.replace(".wav", "")


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
class SaragaDataset(Dataset):
    def __init__(self, tsv_path, processor, label_map=None):
        self.processor = processor
        self.audio_entries = []

        if not os.path.exists(tsv_path):
            raise FileNotFoundError(f"Manifest missing at destination path: {tsv_path}")

        with open(tsv_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
            if len(lines) == 0:
                raise ValueError(f"Target manifest file is completely empty: {tsv_path}")

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
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

        audio = audio[:MAX_SAMPLES]
        waveform = torch.tensor(audio, dtype=torch.float32)
        label_name = get_label_name_from_filename(os.path.basename(file_path))
        label = self.label_map.get(label_name, 0)
        return waveform, label


# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────
class MERTClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.compute_dtype = (
            torch.bfloat16
            if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            else torch.float32
        )
        print(f"Initializing MERT Backbone using Data Type Precision: {self.compute_dtype}")

        self.backbone = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype=self.compute_dtype,
        )

        for param in self.backbone.parameters():
            param.requires_grad = False

        hidden = self.backbone.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = x.to(dtype=self.compute_dtype)
        dev_type = "cuda" if torch.cuda.is_available() else "cpu"

        with torch.autocast(device_type=dev_type, dtype=self.compute_dtype):
            outputs = self.backbone(
                input_values=x,
                output_hidden_states=True,
            )

        embeddings = torch.mean(outputs.last_hidden_state.float(), dim=1)
        logits = self.classifier(embeddings)
        return logits, embeddings


# ──────────────────────────────────────────────
# Gradual Unfreezing Schedule
# ──────────────────────────────────────────────
def update_unfreezing_groups(model, epoch, optimizer, base_lr=1e-4):
    if epoch < 2:
        return

    if not (hasattr(model.backbone, "encoder") and hasattr(model.backbone.encoder, "layers")):
        return

    layers = model.backbone.encoder.layers
    num_layers = len(layers)
    target_layers = []

    if epoch == 2:
        target_layers = list(range(9, num_layers))
        msg = "Layers 9 to 11"
    elif epoch == 4:
        target_layers = list(range(6, 9))
        msg = "Layers 6 to 8"
    elif epoch == 6:
        target_layers = list(range(3, 6))
        msg = "Layers 3 to 5"
    elif epoch == 8:
        target_layers = list(range(0, 3))
        msg = "Entire Backbone Tree"

    if target_layers:
        new_params = []
        for i in target_layers:
            for param in layers[i].parameters():
                if not param.requires_grad:
                    param.requires_grad = True
                    new_params.append(param)

        if new_params:
            optimizer.add_param_group({
                "params": new_params,
                "lr": base_lr * 0.05,
            })
            print(f"  [Schedule] Milestone Achieved: Unfrozen {msg}. Added parameters to optimizer.")


# ──────────────────────────────────────────────
# PANN Manifest Speech Filter Logic
# ──────────────────────────────────────────────
def run_pann_speech_filter(input_tsv, output_tsv, device):
    """Reads a manifest tsv, computes speech probabilities using PANN, and strips speech entries."""
    if not os.path.exists(input_tsv):
        return input_tsv

    if os.path.exists(output_tsv):
        print(f"  [PANN] Clean manifest already exists at: {output_tsv}. Skipping extraction.")
        return output_tsv

    print(f"  [PANN] Filtering speech signatures out from: {os.path.basename(input_tsv)}...")
    at_model = AudioTagging(checkpoint_path=None, device=device)

    with open(input_tsv, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    base_dir = lines[0]
    cleaned_rows = []
    discarded_count = 0

    for line in lines[1:]:
        if "\t" not in line:
            continue
        rel_path, frames = line.split("\t")
        full_path = os.path.join(base_dir, rel_path)

        try:
            # PANN runs natively at 32kHz
            audio, _ = librosa.load(full_path, sr=PANN_SAMPLE_RATE, mono=True)
            audio_tensor = audio[None, :]  # Add batch dimension -> (1, samples)

            clipwise_output, _ = at_model.inference(audio_tensor)

            # Index 0 in AudioSet maps to Speech
            speech_prob = clipwise_output[0][0]

            if speech_prob > SPEECH_THRESHOLD:
                discarded_count += 1
            else:
                cleaned_rows.append(f"{rel_path}\t{frames}")
        except Exception:
            continue

    with open(output_tsv, "w", encoding="utf-8") as f:
        f.write(f"{base_dir}\n")
        for row in cleaned_rows:
            f.write(f"{row}\n")

    print(f"  [PANN] Done! Kept: {len(cleaned_rows)} | Discarded Speech Clutter: {discarded_count}")
    return output_tsv


# ──────────────────────────────────────────────
# Training + Evaluation Pipeline
# ──────────────────────────────────────────────
def run_pipeline(model_name, run_label):
    print(f"\n STARTING PIPELINE RUN: {run_label.upper()} ")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name, trust_remote_code=True)

    def collate_fn(batch):
        waveforms = [wave.numpy() for wave, label in batch]
        labels = [label for wave, label in batch]
        inputs = processor(waveforms, sampling_rate=SAMPLE_RATE, padding=True, return_tensors="pt")
        return inputs.input_values, torch.tensor(labels, dtype=torch.long)

    manifest_base_dir = "./saraga_data/manifests"
    train_tsv = os.path.join(manifest_base_dir, "train.tsv")
    valid_tsv = os.path.join(manifest_base_dir, "valid.tsv")
    test_tsv = os.path.join(manifest_base_dir, "test.tsv")

    # Dynamic PANN Clean Hooks Implementation
    if PANN_AVAILABLE:
        print("\n[PANN Initialization] Executing dataset verification stream...")
        train_tsv = run_pann_speech_filter(train_tsv, os.path.join(manifest_base_dir, "train_clean.tsv"), device)
        valid_tsv = run_pann_speech_filter(valid_tsv, os.path.join(manifest_base_dir, "valid_clean.tsv"), device)
        test_tsv = run_pann_speech_filter(test_tsv, os.path.join(manifest_base_dir, "test_clean.tsv"), device)
    else:
        print("\n[PANN Warning] panns-inference package not detected. Training on raw manifests.")

    temp_train = SaragaDataset(train_tsv, processor)
    temp_valid = SaragaDataset(valid_tsv, processor)
    temp_test = SaragaDataset(test_tsv, processor)

    combined_label_map = {}
    all_entries = temp_train.audio_entries + temp_valid.audio_entries + temp_test.audio_entries
    for path in all_entries:
        label_name = get_label_name_from_filename(os.path.basename(path))
        if label_name not in combined_label_map:
            combined_label_map[label_name] = len(combined_label_map)

    train_ds = SaragaDataset(train_tsv, processor, label_map=combined_label_map)
    valid_ds = SaragaDataset(valid_tsv, processor, label_map=combined_label_map)
    test_ds = SaragaDataset(test_tsv, processor, label_map=combined_label_map)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = MERTClassifier(model_name, num_classes=train_ds.num_classes).to(device)

    base_learning_rate = 3e-4
    optimizer = torch.optim.AdamW(
        [{"params": filter(lambda p: p.requires_grad, model.classifier.parameters()), "lr": base_learning_rate}]
    )
    criterion = nn.CrossEntropyLoss()

    # ── Training Loop ──
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        update_unfreezing_groups(model, epoch, optimizer, base_lr=base_learning_rate)

        model.train()
        optimizer.zero_grad()
        train_loss = 0.0

        for step, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits, _ = model(inputs)
            loss = criterion(logits, labels) / GRAD_ACCUM_STEPS
            loss.backward()
            train_loss += loss.item() * GRAD_ACCUM_STEPS

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            if step % 100 == 0:
                print(f"  Step {step:4d} | Batch Running Loss: {loss.item() * GRAD_ACCUM_STEPS:.4f}")

            del inputs, labels, logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ── Validation ──
        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                logits, _ = model(inputs)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                del inputs, labels, logits

        print(
            f"Epoch {epoch + 1} Summary -> "
            f"Train Loss: {train_loss / len(train_loader):.4f} | "
            f"Valid Loss: {val_loss / len(valid_loader):.4f} | "
            f"Valid Acc: {accuracy_score(val_labels, val_preds):.4f}"
        )

    # ── Test Evaluation ──
    print("\nEvaluating model on final test set split entries...")
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    extracted_embeddings = {}

    with torch.no_grad():
        for step, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            logits, embeddings = model(inputs)
            all_probs.extend(torch.softmax(logits, dim=1).cpu().numpy())
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
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

    print(f"\n--- Final Metrics Summary Result Logs for: {run_label.upper()} ---")
    print(f"Overall Accuracy Score: {accuracy_score(all_labels, all_preds):.4f}")
    if train_ds.num_classes >= 5:
        print(
            f"Top-5 Accuracy Matrix: "
            f"{top_k_accuracy_score(all_labels, np.array(all_probs), k=5, labels=list(range(train_ds.num_classes))):.4f}"
        )
    print(classification_report(all_labels, all_preds, zero_division=0))


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    run_pipeline("ntua-slp/CultureMERT-95M", "cultural_mert_gradual")