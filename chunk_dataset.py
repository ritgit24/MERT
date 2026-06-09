import os
import glob
from pydub import AudioSegment

def chunk_local_saraga(raw_data_dir, output_chunks_dir, chunk_length_ms=30000):
    """
    Scans the local manually downloaded Saraga folder structure and 
    splits all tracks into 16kHz 30-second chunks for MERT training.
    """
    os.makedirs(output_chunks_dir, exist_ok=True)
    
    print("--- STEP 1: SCANNING LOCAL SARAGA TRACKS ---")
    # Scan for any wav or mp3 files inside both carnatic and hindustani subdirectories
    audio_files = glob.glob(os.path.join(raw_data_dir, "**/*.wav"), recursive=True) + \
                  glob.glob(os.path.join(raw_data_dir, "**/*.mp3"), recursive=True)
                  
    print(f"Found {len(audio_files)} local tracks to process.")
    
    for file_path in audio_files:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"Processing track: {base_name} ...")
        
        try:
            audio = AudioSegment.from_file(file_path)
            duration_ms = len(audio)
            
            chunks_created = 0
            for i in range(0, duration_ms, chunk_length_ms):
                chunk = audio[i:i + chunk_length_ms]
                
                # Skip any trailing piece shorter than 30 seconds
                if len(chunk) < chunk_length_ms:
                    continue
                    
                chunk_name = f"{base_name}_chunk_{chunks_created}.wav"
                chunk_output_path = os.path.join(output_chunks_dir, chunk_name)
                
                # Export with standard 16kHz sampling rate required by MERT audio checkpoints
                chunk.export(chunk_output_path, format="wav", parameters=["-ar", "16000"])
                chunks_created += 1
                
            print(f"-> Created {chunks_created} uniform chunks.")
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")
            
    print(f"\nAll done! Processed clips are safely stored in: {output_chunks_dir}")


RAW_DATA_PATH = "./saraga_raw_data"
PROCESSED_CHUNKS_PATH = "./saraga_data/processed_chunks"

if __name__ == "__main__":
    chunk_local_saraga(RAW_DATA_PATH, PROCESSED_CHUNKS_PATH)