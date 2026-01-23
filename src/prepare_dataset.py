import os
import soundfile as sf
import librosa
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import random
import sys
import urllib.request
import zipfile
import shutil

# --- KONFIGURASJON ---
TARGET_SPEAKER = "Anniken Huitfeldt" 

# Stier inne i Docker-containeren
OUTPUT_WAV_DIR = "/app/Data/wavs"
TRAIN_LIST_FILE = "/app/Data/train_list.txt"
VAL_LIST_FILE = "/app/Data/val_list.txt"
UTILS_DIR = "/app/Utils"

# StyleTTS2 standard
SAMPLE_RATE = 24000

def prepare_data():
    print(f"🚀 Starter forberedelse for stemmen: {TARGET_SPEAKER}")
    
    os.makedirs(OUTPUT_WAV_DIR, exist_ok=True)
    os.makedirs("/app/Data", exist_ok=True)
    os.makedirs(UTILS_DIR, exist_ok=True)
    
    print("⏳ Kobler til NPSC-datasettet (Streaming)...")
    
    try:
        ds = load_dataset(
            "NbAiLab/NPSC", 
            "16K_mp3_bokmaal", 
            split="train", 
            streaming=True, 
            trust_remote_code=True
        )
    except Exception as e:
        print(f"❌ Kunne ikke laste datasett: {e}")
        sys.exit(1)
    
    data_entries = []
    total_duration = 0
    processed_count = 0
    
    print("⏳ Laster ned, analyserer og konverterer lydfiler...")
    
    MAX_DURATION_SECONDS = 14400 # 4 timer
    
    for i, row in enumerate(tqdm(ds)):
        if total_duration > MAX_DURATION_SECONDS: 
            break
            
        if row.get("speaker_name") != TARGET_SPEAKER:
            continue
            
        try:
            audio_data = row["audio"]
            audio_array = audio_data["array"]
            orig_sr = audio_data["sampling_rate"]
            duration_seconds = len(audio_array) / orig_sr
        except Exception:
            continue 

        if duration_seconds < 1.5 or duration_seconds > 12.0:
            continue
            
        if orig_sr != SAMPLE_RATE:
            audio_array = librosa.resample(audio_array, orig_sr=orig_sr, target_sr=SAMPLE_RATE)
            
        max_val = np.max(np.abs(audio_array))
        if max_val > 0:
            audio_array = audio_array / max_val
        
        filename = f"{processed_count:05d}.wav"
        filepath = os.path.join(OUTPUT_WAV_DIR, filename)
        sf.write(filepath, audio_array, SAMPLE_RATE)
        
        text = row["text"].replace("|", "").strip()
        entry = f"{filename}|{text}|0"
        data_entries.append(entry)
        
        total_duration += duration_seconds
        processed_count += 1
        
        if processed_count % 50 == 0:
            print(f"   --> Prosessert {processed_count} filer ({total_duration/60:.1f} minutter totalt)")

    if len(data_entries) == 0:
        print(f"❌ FEIL: Fant ingen lydfiler for '{TARGET_SPEAKER}'. Sjekk navnet!")
        sys.exit(1)

    print(f"✅ Ferdig! Totalt {len(data_entries)} klipp ({total_duration/60:.1f} minutter).")
    
    random.shuffle(data_entries)
    split_idx = int(len(data_entries) * 0.95)
    train_data = data_entries[:split_idx]
    val_data = data_entries[split_idx:]
    
    with open(TRAIN_LIST_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(train_data))
        
    with open(VAL_LIST_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(val_data))
        
    print(f"📝 Lagret {len(train_data)} linjer til train_list.txt")
    
    # --- FIX: LAST NED ORIGINAL PL-BERT FRA HUGGING FACE ---
    # Vi bruker yl4579 (skaperen av StyleTTS2) sitt HF repo. Det er trygt.
    
    plbert_dir = os.path.join(UTILS_DIR, "PLBERT")
    os.makedirs(plbert_dir, exist_ok=True)
    
    # Dette er de to filene som lå inni bert.zip
    files_to_download = {
        "config.json": "https://huggingface.co/yl4579/StyleTTS2-LibriTTS/resolve/main/Utils/PLBERT/config.json",
        "step_1000000.t7": "https://huggingface.co/yl4579/StyleTTS2-LibriTTS/resolve/main/Utils/PLBERT/step_1000000.t7"
    }
    
    print("📥 Laster ned PL-BERT filer fra Hugging Face (ingen zip-tull)...")
    
    for filename, url in files_to_download.items():
        file_path = os.path.join(plbert_dir, filename)
        
        # Sjekk størrelse for å se om vi har en korrupt fil (viktig!)
        if os.path.exists(file_path):
            if os.path.getsize(file_path) < 1000: # Hvis filen er mistenkelig liten (f.eks feilmelding)
                print(f"   ⚠️  Filen {filename} ser ødelagt ut. Laster ned på nytt.")
                os.remove(file_path)
        
        if not os.path.exists(file_path):
            print(f"   Laster ned {filename}...")
            try:
                urllib.request.urlretrieve(url, file_path)
            except Exception as e:
                print(f"❌ Feil ved nedlasting av {filename}: {e}")
                sys.exit(1)
        else:
            print(f"   {filename} er allerede på plass.")

    print("✅ PL-BERT installert korrekt!")

if __name__ == "__main__":
    prepare_data()
