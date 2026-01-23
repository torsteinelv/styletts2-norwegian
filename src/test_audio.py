import torch
import yaml
import os
import soundfile as sf
import librosa
import numpy as np
import sys
# Vi må legge til rot-mappen for å finne StyleTTS2-modulene
sys.path.append('/app')

from models import StyleTTS2
from text_utils import TextCleaner
from Utils.PLBERT.util import load_plbert

# --- KONFIGURASJON ---
MODEL_PATH = "Models/NorskStyle/epoch_2nd_00020.pth" # Vi tar epoch 20
CONFIG_PATH = "configs/config_norsk.yml"
OUTPUT_DIR = "samples"
TEXTS = [
    "Dette er en test av den nye norske stemmen.",
    "President, jeg mener at dette forslaget er helt uansvarlig.",
    "Vi må sørge for at alle får være med på utviklingen.",
    "Tusen takk for oppmerksomheten."
]

def generate_samples():
    print("🎵 Starter generering av lydprøver...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Last Config
    config = yaml.safe_load(open(CONFIG_PATH))
    
    # 2. Last Modell
    model = StyleTTS2(config['model_params']).to('cuda')
    
    # Sjekk om modellen finnes
    if not os.path.exists(MODEL_PATH):
        # Hvis epoch 20 ikke finnes, ta den nyeste
        files = [f for f in os.listdir("Models/NorskStyle") if f.endswith(".pth")]
        if not files:
            print("❌ Ingen modell funnet!")
            return
        files.sort()
        MODEL_PATH = os.path.join("Models/NorskStyle", files[-1])
        print(f"⚠️ Fant ikke epoch 20, bruker: {MODEL_PATH}")

    params = torch.load(MODEL_PATH, map_location='cpu')
    model.load_state_dict(params['net'])
    model.eval()
    print("✅ Modell lastet.")

    # 3. Klargjør tekst-prosessering
    text_cleaner = TextCleaner()
    plbert = load_plbert(config['log_dir'], config['plbert_dir'] if 'plbert_dir' in config else 'Utils/PLBERT/')
    
    # 4. Finn en referansestemme (Style)
    # Vi tar en tilfeldig fil fra treningsdataene for å kopiere "stilen" til Anniken
    ref_wavs = [f for f in os.listdir("Data/wavs") if f.endswith(".wav")]
    ref_wav_path = os.path.join("Data/wavs", ref_wavs[0]) # Tar den første
    print(f"Using reference audio: {ref_wav_path}")
    
    # Prosesser referanse-lyd
    ref_audio, _ = librosa.load(ref_wav_path, sr=24000)
    ref_audio = torch.FloatTensor(ref_audio).unsqueeze(0).to('cuda')
    
    # Regn ut Style Vector
    with torch.no_grad():
        style = model.style_encoder(ref_audio)

    # 5. Generer tale
    for i, text in enumerate(TEXTS):
        print(f"🗣️ Sier: '{text}'")
        
        # Tekst til fonemer
        tokens = text_cleaner(text)
        tokens = torch.LongTensor(tokens).unsqueeze(0).to('cuda')
        
        with torch.no_grad():
            output = model.inference(tokens, style, alpha=1.0) # alpha=1.0 er normal fart
            
        audio = output.squeeze().cpu().numpy()
        path = os.path.join(OUTPUT_DIR, f"sample_{i}.wav")
        sf.write(path, audio, 24000)
        print(f"   Lagret til {path}")

if __name__ == "__main__":
    generate_samples()
