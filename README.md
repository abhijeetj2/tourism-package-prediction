# Tourism MLOps Project

This package is ready to upload to a GitHub repository for the "Visit with Us" project.

## What this includes

- End-to-end MLOps pipeline via GitHub Actions
- Data preparation and Hugging Face dataset registration
- Model training, parameter tuning, experiment tracking, and model registration
- Streamlit deployment files for Hugging Face Spaces

## Project Structure

- `.github/workflows/pipeline.yml`
- `tourism_project/data_preparation.py`
- `tourism_project/train_and_register.py`
- `tourism_project/github_actions_requirements.txt`
- `tourism_project/deployment/Dockerfile`
- `tourism_project/deployment/app.py`
- `tourism_project/deployment/requirements.txt`
- `tourism_project/deployment/deploy_to_space.py`
- `tourism_project/data/.gitkeep`



## Required GitHub Secrets

Add these in your repository settings:

- `HF_TOKEN`
- `HF_DATASET_REPO` (example: `your-username/visit-with-us-dataset`)
- `HF_MODEL_REPO` (example: `your-username/visit-with-us-model`)
- `HF_SPACE_REPO` (example: `your-username/visit-with-us-space`)
