import os
import soundfile as sf
import librosa
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import random
import sys

# --- KONFIGURASJON ---
# ENDRE DENNE til navnet på din favoritt-taler!
# Forslag: "Anniken Huitfeldt", "Jonas Gahr Støre", "Erna Solberg", "Trine Skei Grande"
TARGET_SPEAKER = "Anniken Huitfeldt" 

# Stier inne i Docker-containeren
OUTPUT_WAV_DIR = "/app/Data/wavs"
TRAIN_LIST_FILE = "/app/Data/train_list.txt"
VAL_LIST_FILE = "/app/Data/val_list.txt"

# StyleTTS2 standard
SAMPLE_RATE = 24000

def prepare_data():
    print(f"🚀 Starter forberedelse for stemmen: {TARGET_SPEAKER}")
    
    os.makedirs(OUTPUT_WAV_DIR, exist_ok=True)
    
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
        sys.exit(1)
    
    data_entries = []
    total_duration = 0
    processed_count = 0
    
    print("⏳ Laster ned, analyserer og konverterer lydfiler...")
    print("   (Dette kan ta litt tid før fremdriftsbaren starter ordentlig)")

    # Vi setter en maks grense på 2.5 timer med lyd (nok for StyleTTS2)
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
        # StyleTTS2 liker setninger på 2-12 sekunder best.
        if duration_seconds < 1.5 or duration_seconds > 12.0:
            continue
            
        # --- AUDIO PROSESSERING ---
        
        # Resample til 24kHz (StyleTTS2 krav)
        if orig_sr != SAMPLE_RATE:
            audio_array = librosa.resample(audio_array, orig_sr=orig_sr, target_sr=SAMPLE_RATE)
            
        # Normaliser volum (viktig for jevn lyd!)
        # Dette sikrer at lyden ligger mellom -1.0 og 1.0
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
        # Vi bruker speaker_id 0 siden vi trener en single-speaker modell
        entry = f"{filename}|{text}|0"
        data_entries.append(entry)
        
        total_duration += duration_seconds
        processed_count += 1
        
        # Print status hver 50. fil så du ser at det skjer noe i loggen
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

    # --- LAST NED HJELPEMODELL (PL-BERT) ---
    print("📥 Laster ned PL-BERT (viktig for StyleTTS2)...")
    # Vi sjekker om den allerede finnes for å spare tid
    if not os.path.exists("Utils/PLBERT/config.json"):
        os.system("wget -q https://github.com/yl4579/StyleTTS2/releases/download/v1.0/bert.zip")
        os.system("unzip -o -q bert.zip -d Utils/")
        os.system("rm bert.zip")
        print("✅ PL-BERT installert.")
    else:
        print("✅ PL-BERT fantes allerede.")

if __name__ == "__main__":
    prepare_data()
