import os
from huggingface_hub import HfApi, login

def upload_model():
    print("☁️ Laster opp modell til Hugging Face...")
    
    hf_token = os.getenv("HF_TOKEN")
    hf_repo = os.getenv("HF_REPO")
    
    if not hf_token or not hf_repo:
        print("❌ Mangler HF_TOKEN eller HF_REPO")
        return

    login(token=hf_token)
    api = HfApi()
    
    MODEL_DIR = "Models/NorskStyle"
    
    # Finn siste epoch (beste modell)
    checkpoints = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pth")]
    if not checkpoints:
        print("❌ Ingen modellfiler funnet!")
        return
        
    # Vi laster opp hele mappen
    print(f"Laster opp {MODEL_DIR} til {hf_repo}...")
    
    api.upload_folder(
        folder_path=MODEL_DIR,
        repo_id=hf_repo,
        commit_message="Upload Norwegian StyleTTS2 Model"
    )
    
    print("🎉 Opplasting ferdig!")

if __name__ == "__main__":
    upload_model()
