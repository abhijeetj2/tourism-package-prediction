import os
from huggingface_hub import HfApi

HF_TOKEN = os.getenv("HF_TOKEN")
HF_SPACE_REPO = os.getenv("HF_SPACE_REPO", "")

if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN is required for deployment.")
if not HF_SPACE_REPO:
    raise EnvironmentError("HF_SPACE_REPO is required, e.g., username/visit-with-us-space")

api = HfApi(token=HF_TOKEN)
api.create_repo(repo_id=HF_SPACE_REPO, repo_type="space", space_sdk="docker", exist_ok=True)

api.upload_folder(
    folder_path="tourism_project/deployment",
    repo_id=HF_SPACE_REPO,
    repo_type="space",
)

print(f"Deployment files pushed to: https://huggingface.co/spaces/{HF_SPACE_REPO}")
