import os
import soundfile as sf
import librosa
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import random

# --- KONFIGURASJON ---
# ENDRE DENNE til navnet på din favoritt-taler!
# Kjør 'find_voice.py' først for å finne navnet hvis du er usikker.
TARGET_SPEAKER = "Anniken Huitfeldt" 

OUTPUT_WAV_DIR = "/app/Data/wavs"
TRAIN_LIST_FILE = "/app/Data/train_list.txt"
VAL_LIST_FILE = "/app/Data/val_list.txt"
SAMPLE_RATE = 24000

def prepare_data():
    print(f"🚀 Starter forberedelse for stemmen: {TARGET_SPEAKER}")
    
    os.makedirs(OUTPUT_WAV_DIR, exist_ok=True)
    
    # Last datasett (streamer for hastighet)
    ds = load_dataset(
        "NbAiLab/NPSC", 
        "16K_mp3_bokmaal", 
        split="train", 
        streaming=True, 
        trust_remote_code=True  # <--- LEGG TIL DENNE!
    )
    
    data_entries = []
    total_duration = 0
    
    print("⏳ Laster ned og konverterer lydfiler...")
    
    # Gå gjennom datasettet
    for i, row in enumerate(tqdm(ds)):
        # Stopp etter 2 timer med lyd (nok for StyleTTS2)
        if total_duration > 7200: 
            break
            
        if row["speaker_name"] != TARGET_SPEAKER:
            continue
            
        # Filtrer vekk veldig korte klipp (< 1 sek) og veldig lange (> 15 sek)
        duration = row["duration"]
        if duration < 1.0 or duration > 15.0:
            continue
            
        # Prosesser lyd
        audio_array = row["audio"]["array"]
        orig_sr = row["audio"]["sampling_rate"]
        
        # Resample til 24kHz
        if orig_sr != SAMPLE_RATE:
            audio_array = librosa.resample(audio_array, orig_sr=orig_sr, target_sr=SAMPLE_RATE)
            
        # Normaliser volum
        audio_array = audio_array / np.max(np.abs(audio_array))
        
        # Lagre fil
        filename = f"{i:05d}.wav"
        filepath = os.path.join(OUTPUT_WAV_DIR, filename)
        sf.write(filepath, audio_array, SAMPLE_RATE)
        
        # Rensk tekst (fjerne rare tegn)
        text = row["text"].replace("|", "")
        
        # Formatet StyleTTS2 vil ha: "filename.wav|Text content|SpeakerID"
        entry = f"{filename}|{text}|0"
        data_entries.append(entry)
        
        total_duration += duration

    print(f"✅ Fant {len(data_entries)} klipp ({total_duration/60:.1f} minutter).")
    
    # Split Train/Val (95% trening, 5% test)
    random.shuffle(data_entries)
    split_idx = int(len(data_entries) * 0.95)
    train_data = data_entries[:split_idx]
    val_data = data_entries[split_idx:]
    
    with open(TRAIN_LIST_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(train_data))
        
    with open(VAL_LIST_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(val_data))
        
    print(f"📝 Lagret {len(train_data)} linjer til train_list.txt")
    print(f"📝 Lagret {len(val_data)} linjer til val_list.txt")

    # Last ned PL-BERT (Hjelpemodell for fonemer)
    print("📥 Laster ned PL-BERT...")
    os.system("wget https://github.com/yl4579/StyleTTS2/releases/download/v1.0/bert.zip")
    os.system("unzip -o bert.zip -d Utils/")
    os.system("rm bert.zip")

if __name__ == "__main__":
    prepare_data()
