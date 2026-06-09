import os
import glob
import torch
import torchaudio
import numpy as np
import pandas as pd
from transformers import Wav2Vec2FeatureExtractor, AutoModel

def extract_mert_embeddings(model_name, audio_dir, output_npy_path):
    """
    Loads a MERT variant from Hugging Face, processes local audio chunks,
    and extracts the final hidden-state embeddings.
    """
    print(f"\n--- Loading Model: {model_name} ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the specialized audio feature processor and backbone neural net
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    model.eval()

    audio_files = glob.glob(os.path.join(audio_dir, "*.wav"))
    print(f"Found {len(audio_files)} chunks for feature extraction.")

    embedding_dict = {}

    with torch.no_grad():
        for i, file_path in enumerate(audio_files):
            filename = os.path.basename(file_path)
            try:
                # Load audio chunk (ensuring it matches MERT's expected 16kHz rate)
                waveform, sr = torchaudio.load(file_path)
                if sr != 16000:
                    resampler = torchaudio.transforms.Resample(sr, 16000)
                    waveform = resampler(waveform)
                
                # Convert matrix channels to mono if necessary
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)

                # Format inputs into tensors for the model
                inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
                input_values = inputs.input_values.to(device)

                # Extract hidden representations
                outputs = model(input_values=input_values, output_hidden_states=True)
                
                # Mean-pool across the time dimension to get a single vector per audio chunk
                # Layer -1 yields the final developed representation space
                last_hidden_state = outputs.last_hidden_state  # Shape: [1, Time, Hidden_Dim]
                pooled_embedding = torch.mean(last_hidden_state, dim=1).squeeze().cpu().numpy()

                embedding_dict[filename] = pooled_embedding
                
                if (i + 1) % 50 == 0:
                    print(f"Processed {i + 1}/{len(audio_files)} tracks...")
            except Exception as e:
                print(f"Error extracting from {filename}: {e}")

    # Save output matrix vector files dictionary to disk
    np.save(output_npy_path, embedding_dict)
    print(f"Successfully saved embeddings to {output_npy_path}")

if __name__ == "__main__":
    CHUNKS_DIR = "./saraga_data/processed_chunks"
    
    # Run A: Standard Base MERT
    extract_mert_embeddings(
        model_name="m-a-p/MERT-95M", 
        audio_dir=CHUNKS_DIR, 
        output_npy_path="./mert_base_embeddings.npy"
    )
    
    # Run B: Cultural MERT (Replace with the exact Hugging Face path provided by your senior)
    # Example placeholder: "m-a-p/MERT-95M-public" or your local checkpoint path string
    extract_mert_embeddings(
        model_name="m-a-p/MERT-95M", # Update to your specific Cultural MERT variant path
        audio_dir=CHUNKS_DIR, 
        output_npy_path="./cultural_mert_embeddings.npy"
    )
