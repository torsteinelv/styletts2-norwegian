import os
from huggingface_hub import HfApi, login

def upload_model():
    print("☁️ Laster opp modell OG prøver til Hugging Face...")
    
    hf_token = os.getenv("HF_TOKEN")
    hf_repo = os.getenv("HF_REPO")
    
    if not hf_token or not hf_repo:
        print("❌ Mangler HF_TOKEN eller HF_REPO")
        return

    login(token=hf_token)
    api = HfApi()
    
    # 1. Last opp Modellen (Selve hjernen)
    MODEL_DIR = "Models/NorskStyle"
    print(f"Laster opp modell fra {MODEL_DIR}...")
    api.upload_folder(
        folder_path=MODEL_DIR,
        repo_id=hf_repo,
        commit_message="Upload Trained Model"
    )
    
    # 2. Last opp Lydprøver (Beviset)
    SAMPLES_DIR = "samples"
    if os.path.exists(SAMPLES_DIR):
        print(f"Laster opp lydprøver fra {SAMPLES_DIR}...")
        api.upload_folder(
            folder_path=SAMPLES_DIR,
            path_in_repo="samples", # Legg dem i en undermappe på HF
            repo_id=hf_repo,
            commit_message="Add audio samples"
        )
    
    print("🎉 Alt lastet opp! Sjekk 'samples'-mappen på Hugging Face.")

if __name__ == "__main__":
    upload_model()
