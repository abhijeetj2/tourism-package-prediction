import json
import os
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path("tourism_project")
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model_building"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "")
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "")
api = HfApi(token=HF_TOKEN) if HF_TOKEN else None

if api and HF_DATASET_REPO:
    train_file = hf_hub_download(repo_id=HF_DATASET_REPO, filename="processed/train.csv", repo_type="dataset", token=HF_TOKEN)
    test_file = hf_hub_download(repo_id=HF_DATASET_REPO, filename="processed/test.csv", repo_type="dataset", token=HF_TOKEN)
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
else:
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

X_train = train_df.drop(columns=["ProdTaken"])
y_train = train_df["ProdTaken"]
X_test = test_df.drop(columns=["ProdTaken"])
y_test = test_df["ProdTaken"]

num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(random_state=42)),
])

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 8, 12],
    "model__min_samples_split": [2, 5],
}

mlflow.set_tracking_uri(f"file:{(MODEL_DIR / 'mlruns').resolve()}")
mlflow.set_experiment("tourism-wellness-package")

with mlflow.start_run(run_name="rf-gridsearch"):
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    for k, v in grid.best_params_.items():
        mlflow.log_param(k, v)
    for k, v in metrics.items():
        mlflow.log_metric(k, v)

    cv_results = pd.DataFrame(grid.cv_results_)
    cv_results_path = MODEL_DIR / "cv_results.csv"
    cv_results.to_csv(cv_results_path, index=False)
    mlflow.log_artifact(str(cv_results_path))

    model_path = MODEL_DIR / "best_model.joblib"
    metrics_path = MODEL_DIR / "metrics.json"
    joblib.dump(best_model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

if api and HF_MODEL_REPO:
    api.create_repo(repo_id=HF_MODEL_REPO, repo_type="model", exist_ok=True)
    api.upload_file(path_or_fileobj=str(model_path), path_in_repo="best_model.joblib", repo_id=HF_MODEL_REPO, repo_type="model")
    api.upload_file(path_or_fileobj=str(metrics_path), path_in_repo="metrics.json", repo_id=HF_MODEL_REPO, repo_type="model")
    api.upload_file(path_or_fileobj=str(cv_results_path), path_in_repo="cv_results.csv", repo_id=HF_MODEL_REPO, repo_type="model")

print("Training and registration complete.")
