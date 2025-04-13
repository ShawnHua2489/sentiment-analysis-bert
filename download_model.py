from huggingface_hub import snapshot_download
from config import Config

def download_model():
    print("Downloading model files using snapshot_download...")
    try:
        # Download the model files
        snapshot_download(
            repo_id=Config.MODEL_NAME,
            local_dir="models/distilbert",
            local_dir_use_symlinks=False
        )
        print("Model files downloaded successfully!")
        
    except Exception as e:
        print(f"Error downloading model: {e}")

if __name__ == "__main__":
    download_model() 