import os
import soundfile as sf
import librosa
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import random
import sys
import shutil
import yaml
import json
from huggingface_hub import hf_hub_download

# --- KONFIGURASJON ---
TARGET_SPEAKER = "Anniken Huitfeldt" 

# Stier inne i Docker-containeren
OUTPUT_WAV_DIR = "/app/Data/wavs"
TRAIN_LIST_FILE = "/app/Data/train_list.txt"
VAL_LIST_FILE = "/app/Data/val_list.txt"
UTILS_DIR = "/app/Utils"
TEXT_UTILS_FILE = "/app/text_utils.py"  # Filen som styrer uttale

# StyleTTS2 standard
SAMPLE_RATE = 24000

def patch_text_utils_to_norwegian():
    """
    Dette er den viktige hacken!
    Vi endrer kildekoden til StyleTTS2 slik at den bruker NORSK lydskrift (phonemizer),
    ikke engelsk. Uten dette høres norsken ut som engelsk gibberish.
    """
    print("🇳🇴 Patcher text_utils.py til å bruke norsk språk...")
    
    if not os.path.exists(TEXT_UTILS_FILE):
        print(f"⚠️ Fant ikke {TEXT_UTILS_FILE}, kan ikke patche språk!")
        return

    try:
        with open(TEXT_UTILS_FILE, 'r') as f:
            code = f.read()
        
        # Sjekk om den allerede er patchet
        if "language='nb'" in code:
            print("   Allerede patchet til norsk.")
            return

        # Bytt ut engelsk standard med norsk bokmål ('nb')
        # Vi ser etter phonemize-funksjonen som vanligvis har language='en-us'
        new_code = code.replace("language='en-us'", "language='nb'")
        
        # Hvis koden bruker 'en', bytt den også
        new_code = new_code.replace("language='en'", "language='nb'")

        with open(TEXT_UTILS_FILE, 'w') as f:
            f.write(new_code)
            
        print("✅ Suksess! Modellen vil nå snakke norsk (nb).")
        
    except Exception as e:
        print(f"❌ Feil under patching av norsk språk: {e}")

def prepare_data():
    print(f"🚀 Starter forberedelse for stemmen: {TARGET_SPEAKER}")
    
    # 1. Kjør språk-patchen først av alt
    patch_text_utils_to_norwegian()
    
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
    
    # Øker til 6 timer max for å være sikker på at vi får nok
    MAX_DURATION_SECONDS = 21600 
    
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
    
    # --- LAST NED PAPERCUP-MODELL (Multilingual) ---
    plbert_dir = os.path.join(UTILS_DIR, "PLBERT")
    os.makedirs(plbert_dir, exist_ok=True)
    
    print("📥 Laster ned Papercup Multilingual PL-BERT (Norsk støtte)...")
    
    try:
        # Config (YAML -> JSON konvertering for sikkerhets skyld)
        yml_path = hf_hub_download(repo_id="papercup-ai/multilingual-pl-bert", filename="config.yml")
        local_yml_path = os.path.join(plbert_dir, "config.yml")
        shutil.copy(yml_path, local_yml_path)

        with open(local_yml_path, 'r') as f_yml:
            config_data = yaml.safe_load(f_yml)
        with open(os.path.join(plbert_dir, "config.json"), 'w') as f_json:
            json.dump(config_data, f_json, indent=4)
        
        # Token Maps (KRITISK)
        token_path = hf_hub_download(repo_id="papercup-ai/multilingual-pl-bert", filename="token_maps.pkl")
        shutil.copy(token_path, os.path.join(plbert_dir, "token_maps.pkl"))

        # Model Checkpoint (rename til det koden forventer)
        model_path = hf_hub_download(repo_id="papercup-ai/multilingual-pl-bert", filename="step_1100000.t7")
        shutil.copy(model_path, os.path.join(plbert_dir, "step_1000000.t7"))
        
        # Util.py (Fix)
        util_path = hf_hub_download(repo_id="papercup-ai/multilingual-pl-bert", filename="util.py")
        shutil.copy(util_path, os.path.join(plbert_dir, "util.py"))

        print("✅ PL-BERT (Papercup Multilingual) installert korrekt!")
        
    except Exception as e:
        print(f"❌ Feil ved nedlasting fra Hugging Face: {e}")
        sys.exit(1)

if __name__ == "__main__":
    prepare_data()
