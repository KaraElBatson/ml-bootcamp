# Machine Learning Bootcamp Project: Credit Scoring

This repository is a complete, professional-grade template for a Machine Learning Bootcamp project, focusing on a Credit Scoring classification problem. It includes:

- Industry & Problem: Credit scoring (binary classification)
- Dataset Selection: Instructions to get real data (from Kaggle) or generate synthetic sample data
- Repository Structure: Clear directories, working scripts, notebooks, and API for inference

## Project Structure

```
ml-bootcamp/
├─ assets/
│  ├─ images/
│  └─ models/
├─ data/
│  ├─ external/
│  ├─ processed/
│  └─ raw/
├─ docs/
├─ notebooks/
│  ├─ 01_Exploratory_Data_Analysis.ipynb
│  ├─ 02_Baseline_Model.ipynb
│  ├─ 03_Feature_Engineering.ipynb
│  ├─ 04_Model_Optimization.ipynb
│  ├─ 05_Model_Evaluation.ipynb
│  └─ 06_Final_Pipeline.ipynb
├─ scripts/
│  ├─ app.py
│  ├─ config.py
│  ├─ data_loader.py
│  ├─ feature_engineering.py
│  ├─ generate_sample_data.py
│  ├─ inference.py
│  ├─ model_utils.py
│  └─ pipeline.py
├─ .gitignore
├─ requirements.txt
├─ setup.py
└─ README.md
```

## Getting Started

1) Create and activate a Python virtual environment
- Windows (PowerShell):
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
- macOS/Linux:
```
python3 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies
```
pip install -r requirements.txt
python -m ipykernel install --user --name ml-bootcamp --display-name "Python (ml-bootcamp)"
```

3) Data: Use Kaggle or synthetic data
- Option A: Kaggle datasets suitable for credit scoring/classification
  - Give Me Some Credit: https://www.kaggle.com/c/GiveMeSomeCredit
  - Home Credit Default Risk: https://www.kaggle.com/c/home-credit-default-risk
  - Lending Club loan data (hosted mirrors on Kaggle)
- Option B: Generate synthetic sample data
```
python scripts/generate_sample_data.py
```
This writes `data/raw/credit_scoring_data.csv`.

4) Run the full pipeline
```
python scripts/pipeline.py --generate
```
This will generate data (if missing), train multiple models (RF/LGBM/XGB), evaluate them, and save the best pipeline to `assets/models/credit_scoring_model.pkl`.

5) Batch/Single inference from CLI
- Single (interactive):
```
python scripts/inference.py --input single
```
- Batch (CSV):
```
python scripts/inference.py --input path/to/file.csv --output predictions.csv
```

6) Serve a web API
```
uvicorn scripts.app:app --host 0.0.0.0 --port 8000 --reload
```
- Docs: http://127.0.0.1:8000/docs

## Notebooks Overview
- 01_Exploratory_Data_Analysis: EDA, distributions, correlations
- 02_Baseline_Model: Baselines with Logistic Regression and Random Forest
- 03_Feature_Engineering: Illustrates transformations and validation
- 04_Model_Optimization: Hyperparameter tuning examples
- 05_Model_Evaluation: Aggregated metrics, ROC/PR curves
- 06_Final_Pipeline: End-to-end training to model artifact

## Configuration
All tunables are in `scripts/config.py`, including paths, model hyperparameters, evaluation metrics, and API settings.

## Industry & Problem Selection
This template is for credit scoring. To adapt for other problems:
- Change `MODEL_CONFIG['target_column']`
- Adjust features in synthetic generator or your real dataset schema
- Update metrics and scoring as needed

## Testing
You can add tests with `pytest` later. Example commands:
```
pytest -q
pytest --cov=scripts -q
```

## GitHub Setup and Commands
1) Initialize git and make first commit
```
git init
git add .
git commit -m "Initial commit: ML Bootcamp project"
```
2) Create a GitHub repository
- Go to https://github.com/new
- Repository name: ml-bootcamp (or your choice)
- Keep it public or private as desired, without adding a README/.gitignore (you already have them)

3) Add remote and push
Replace YOUR_USERNAME with your GitHub username.
```
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ml-bootcamp.git
git push -u origin main
```
If prompted for auth, use your GitHub credentials or a Personal Access Token.

## Contributing
- Use feature branches and pull requests.
- Keep notebooks clean; commit as HTML or with cleared outputs when possible.

## License
MIT License. Update as needed.
