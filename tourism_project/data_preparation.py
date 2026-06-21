import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, hf_hub_download

BASE_DIR = Path("tourism_project")
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "")

raw_csv_path = DATA_DIR / "tourism.csv"
if not raw_csv_path.exists() and Path("tourism.csv").exists():
    raw_csv_path.write_bytes(Path("tourism.csv").read_bytes())

if not raw_csv_path.exists():
    raise FileNotFoundError("tourism.csv is required")

api = HfApi(token=HF_TOKEN) if HF_TOKEN else None
if api and HF_DATASET_REPO:
    api.create_repo(repo_id=HF_DATASET_REPO, repo_type="dataset", exist_ok=True)
    api.upload_file(path_or_fileobj=str(raw_csv_path), path_in_repo="raw/tourism.csv", repo_id=HF_DATASET_REPO, repo_type="dataset")
    downloaded_raw = hf_hub_download(repo_id=HF_DATASET_REPO, filename="raw/tourism.csv", repo_type="dataset", token=HF_TOKEN)
    df = pd.read_csv(downloaded_raw)
else:
    df = pd.read_csv(raw_csv_path)

clean_df = df.copy()
clean_df.columns = [c.strip() for c in clean_df.columns]
if "CustomerID" in clean_df.columns:
    clean_df = clean_df.drop(columns=["CustomerID"])
clean_df = clean_df.drop_duplicates().reset_index(drop=True)

num_cols = clean_df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = clean_df.select_dtypes(exclude=[np.number]).columns.tolist()
for c in num_cols:
    clean_df[c] = clean_df[c].fillna(clean_df[c].median())
for c in cat_cols:
    mode_vals = clean_df[c].mode()
    clean_df[c] = clean_df[c].fillna(mode_vals.iloc[0] if not mode_vals.empty else "Unknown")

train_df, test_df = train_test_split(clean_df, test_size=0.2, random_state=42, stratify=clean_df["ProdTaken"])
train_path = DATA_DIR / "train.csv"
test_path = DATA_DIR / "test.csv"
train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

if api and HF_DATASET_REPO:
    api.upload_file(path_or_fileobj=str(train_path), path_in_repo="processed/train.csv", repo_id=HF_DATASET_REPO, repo_type="dataset")
    api.upload_file(path_or_fileobj=str(test_path), path_in_repo="processed/test.csv", repo_id=HF_DATASET_REPO, repo_type="dataset")

print("Data preparation complete.")
