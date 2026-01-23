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
# Endre denne hvis du vil bytte person.
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
    
    # Lag mapper hvis de ikke finnes
    os.makedirs(OUTPUT_WAV_DIR, exist_ok=True)
    os.makedirs("/app/Data", exist_ok=True)
    os.makedirs(UTILS_DIR, exist_ok=True)
    
    print("⏳ Kobler til NPSC-datasettet (Streaming)...")
    
    try:
        # trust_remote_code=True er nødvendig for NPSC
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

    # Vi setter en maks grense (sekunder) for å ikke fylle disken helt
    MAX_DURATION_SECONDS = 14400 # 4 timer (mer enn nok)
    
    for i, row in enumerate(tqdm(ds)):
        if total_duration > MAX_DURATION_SECONDS: 
            print("🛑 Har nok data! Stopper nedlasting.")
            break
            
        # Sjekk om det er riktig person
        if row.get("speaker_name") != TARGET_SPEAKER:
            continue
            
        # FIX: Manuell utregning av lengde (unngår KeyError: 'duration')
        try:
            audio_data = row["audio"]
            audio_array = audio_data["array"]
            orig_sr = audio_data["sampling_rate"]
            
            # Antall samples / Samplerate = Sekunder
            duration_seconds = len(audio_array) / orig_sr
            
        except Exception as e:
            continue # Hopp over korrupte rader

        # Filtrer vekk støy (veldig korte) og monologer (veldig lange)
        # StyleTTS2 trener best på klipp mellom 1.5 og 12 sekunder.
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
    
    # --- LAST NED PL-BERT (MED USER-AGENT FIX) ---
    bert_config = os.path.join(UTILS_DIR, "PLBERT", "config.json")
    
    if not os.path.exists(bert_config):
        print("📥 Laster ned PL-BERT (viktig for StyleTTS2)...")
        url = "https://github.com/yl4579/StyleTTS2/releases/download/v1.0/bert.zip"
        zip_path = "bert.zip"
        
        try:
            # Fake en User-Agent for å unngå HTTP 404 fra GitHub
            req = urllib.request.Request(
                url, 
                data=None, 
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
            )
            
            print(f"   Laster ned fra {url}...")
            with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
            
            print("   Pakker ut...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(UTILS_DIR)
                
            if os.path.exists(zip_path):
                os.remove(zip_path)
                
            print("✅ PL-BERT installert korrekt!")
            
        except Exception as e:
            print(f"❌ Feil under nedlasting av BERT: {e}")
            sys.exit(1)
    else:
        print("✅ PL-BERT fantes allerede.")

if __name__ == "__main__":
    prepare_data()
