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
# ENDRE DENNE hvis du vil bytte person.
# Gode alternativer: "Anniken Huitfeldt", "Jonas Gahr Støre", "Erna Solberg"
TARGET_SPEAKER = "Anniken Huitfeldt" 

# Stier inne i Docker-containeren
OUTPUT_WAV_DIR = "/app/Data/wavs"
TRAIN_LIST_FILE = "/app/Data/train_list.txt"
VAL_LIST_FILE = "/app/Data/val_list.txt"

# StyleTTS2 standard
SAMPLE_RATE = 24000

def prepare_data():
    print(f"🚀 Starter forberedelse for stemmen: {TARGET_SPEAKER}")
    
    # Lag mapper hvis de ikke finnes
    os.makedirs(OUTPUT_WAV_DIR, exist_ok=True)
    os.makedirs("/app/Data", exist_ok=True)
    
    print("⏳ Kobler til NPSC-datasettet (Streaming)...")
    
    # FIX 1: Legg til trust_remote_code=True for å tillate NPSC-scriptet å kjøre
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
        print("   Tips: Sjekk at 'datasets==2.19.0' er i requirements.txt")
        sys.exit(1)
    
    data_entries = []
    total_duration = 0
    processed_count = 0
    
    print("⏳ Laster ned, analyserer og konverterer lydfiler...")
    print("   (Dette kan ta litt tid før fremdriftsbaren starter ordentlig)")

    # Vi setter en maks grense på ca 2.5 timer med lyd (nok for finetuning)
    MAX_DURATION_SECONDS = 9000 
    
    for i, row in enumerate(tqdm(ds)):
        if total_duration > MAX_DURATION_SECONDS: 
            print("🛑 Har nok data! Stopper nedlasting.")
            break
            
        # Sjekk om det er riktig person
        if row.get("speaker_name") != TARGET_SPEAKER:
            continue
            
        # FIX 2: Manuell utregning av lengde (fikser KeyError: 'duration')
        try:
            audio_data = row["audio"]
            audio_array = audio_data["array"]
            orig_sr = audio_data["sampling_rate"]
            
            # Antall samples / Samplerate = Sekunder
            duration_seconds = len(audio_array) / orig_sr
            
        except Exception as e:
            # Hvis lydfilen er korrupt, hopp over uten å krasje
            continue

        # Filtrer vekk støy (veldig korte) og monologer (veldig lange)
        # StyleTTS2 liker setninger på 1.5 - 12 sekunder best.
        if duration_seconds < 1.5 or duration_seconds > 12.0:
            continue
            
        # --- AUDIO PROSESSERING ---
        
        # Resample til 24kHz (StyleTTS2 krav)
        if orig_sr != SAMPLE_RATE:
            audio_array = librosa.resample(audio_array, orig_sr=orig_sr, target_sr=SAMPLE_RATE)
            
        # Normaliser volum (viktig for jevn lyd!)
        max_val = np.max(np.abs(audio_array))
        if max_val > 0:
            audio_array = audio_array / max_val
        
        # Lagre fil
        filename = f"{processed_count:05d}.wav"
        filepath = os.path.join(OUTPUT_WAV_DIR, filename)
        sf.write(filepath, audio_array, SAMPLE_RATE)
        
        # --- TEKST PROSESSERING ---
        # Fjern '|' siden StyleTTS2 bruker det som separator
        text = row["text"].replace("|", "").strip()
        
        # Format: "filnavn.wav | tekst | speaker_id"
        entry = f"{filename}|{text}|0"
        data_entries.append(entry)
        
        total_duration += duration_seconds
        processed_count += 1
        
        # Print status hver 50. fil
        if processed_count % 50 == 0:
            print(f"   --> Prosessert {processed_count} filer ({total_duration/60:.1f} minutter totalt)")

    if len(data_entries) == 0:
        print(f"❌ FEIL: Fant ingen lydfiler for '{TARGET_SPEAKER}'. Sjekk navnet!")
        sys.exit(1)

    print(f"✅ Ferdig! Totalt {len(data_entries)} klipp ({total_duration/60:.1f} minutter).")
    
    # --- DATA SPLIT (Train/Val) ---
    print("✂️ Splitter i Trening og Validering...")
    random.shuffle(data_entries)
    
    # 95% trening, 5% validering
    split_idx = int(len(data_entries) * 0.95)
    train_data = data_entries[:split_idx]
    val_data = data_entries[split_idx:]
    
    with open(TRAIN_LIST_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(train_data))
        
    with open(VAL_LIST_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(val_data))
        
    print(f"📝 Lagret {len(train_data)} linjer til train_list.txt")
    print(f"📝 Lagret {len(val_data)} linjer til val_list.txt")

    # --- FIX 3: LAST NED PL-BERT (PYTHON NATIVE) ---
    # Vi bruker Python biblioteker i stedet for os.system/wget for å være trygge
    bert_dir = "/app/Utils/PLBERT"
    bert_config = os.path.join(bert_dir, "config.json")
    
    if not os.path.exists(bert_config):
        print("📥 Laster ned PL-BERT (viktig for StyleTTS2)...")
        url = "https://github.com/yl4579/StyleTTS2/releases/download/v1.0/bert.zip"
        zip_path = "bert.zip"
        
        try:
            # Sørg for at mappen Utils eksisterer
            os.makedirs("/app/Utils", exist_ok=True)
            
            # 1. Last ned
            print(f"   Laster ned fra {url}...")
            urllib.request.urlretrieve(url, zip_path)
            
            # 2. Pakk ut
            print("   Pakker ut...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall("/app/Utils")
                
            # 3. Rydd opp
            if os.path.exists(zip_path):
                os.remove(zip_path)
                
            print("✅ PL-BERT installert korrekt!")
            
        except Exception as e:
            print(f"❌ Feil under nedlasting av BERT: {e}")
            # Vi stopper ikke her, i tilfelle filene faktisk ble pakket ut delvis
    else:
        print("✅ PL-BERT fantes allerede.")

if __name__ == "__main__":
    prepare_data()
