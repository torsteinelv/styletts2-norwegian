import os
import subprocess

def run_training():
    print("🔥 Starter trening av Norsk StyleTTS2...")
    
    # Sjekk om data finnes
    if not os.path.exists("/app/Data/train_list.txt"):
        print("❌ Data mangler! Kjører data-preparering først...")
        subprocess.run(["python", "src/prepare_dataset.py"])
        
    # Kommando for å starte trening (bruker original StyleTTS2 train_finetune.py)
    # Vi bruker akselerert trening (DDP) hvis mulig
    cmd = [
        "python", 
        "train_finetune.py", 
        "--config_path", "configs/config_norsk.yml"
    ]
    
    print(f"Kjører kommando: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    run_training()
