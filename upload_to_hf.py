import os
import sys
from huggingface_hub import HfApi

def upload():
    token = os.getenv("HF_TOKEN")
    if not token and len(sys.argv) > 1:
        token = sys.argv[1].strip()
    
    if not token:
        token = input("Please enter your Hugging Face Write Token (hf_...): ").strip()

    if not token:
        print("[Error] No Hugging Face token provided. Exiting.")
        sys.exit(1)

    repo_id = "yashupadhyay/invoice-qwen-vl-lora"
    local_dir = os.path.dirname(os.path.abspath(__file__))

    print("\n========================================================")
    print("      Uploading Model to Hugging Face Hub              ")
    print("========================================================")
    print(f"Target Repository : {repo_id}")
    print(f"Source Directory  : {local_dir}")
    print("--------------------------------------------------------")

    try:
        api = HfApi(token=token)
        user_info = api.whoami()
        username = user_info.get("name")
        print(f"[1/3] Authenticated successfully as: {username}")
        
        # If logged in username differs from yashupadhyay, target user's namespace
        target_repo = f"{username}/invoice-qwen-vl-lora"
        print(f"[2/3] Ensuring repository exists at https://huggingface.co/{target_repo}...")
        api.create_repo(repo_id=target_repo, repo_type="model", exist_ok=True)

        print(f"[3/3] Uploading all model weights and configuration files...")
        api.upload_folder(
            folder_path=local_dir,
            repo_id=target_repo,
            repo_type="model",
            ignore_patterns=[".git", ".git/**", "*.bat", "*.ps1", "upload_to_hf.py"]
        )

        print("\n========================================================")
        print("  UPLOAD COMPLETE! Model is published and live at:")
        print(f"  https://huggingface.co/{target_repo}")
        print("========================================================\n")

    except Exception as e:
        print(f"\n[Fatal Error during upload]: {e}")
        sys.exit(1)

if __name__ == "__main__":
    upload()
