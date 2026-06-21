# Placement Predictor

A full-stack ML project that predicts a student's campus placement probability from their academic record, experience, and assessment scores — with a clean, evaluation-transparent web UI.

Built as a portfolio project to demonstrate end-to-end ML engineering: data pipeline, model comparison, a serving API, and a production-quality frontend — not just a notebook.

## What it does

- Takes a candidate profile (CGPA, internships, projects, DSA score, communication score, branch, backlogs, etc.)
- Returns a placement probability, a binary prediction, and a verdict band
- Shows full model transparency: accuracy/precision/recall/F1/ROC-AUC, a confusion matrix, ROC curve, feature importance, and a 5-fold cross-validation comparison across three candidate models

## Why this is portfolio-worthy (not a toy)

- **Model selection, not model guessing**: three models (Logistic Regression, Random Forest, Gradient Boosting) are compared with stratified 5-fold cross-validation on ROC-AUC, and the winner is picked programmatically — not assumed.
- **Honest evaluation**: held-out test set with accuracy, precision, recall, F1, and ROC-AUC, plus a confusion matrix. The model hits ~77% accuracy / 0.85 ROC-AUC — strong but not suspiciously perfect, which is the right red flag to be aware of for an interview discussion (100% accuracy usually means leakage).
- **Real preprocessing pipeline**: `sklearn.Pipeline` + `ColumnTransformer` for scaling numeric features and one-hot encoding categoricals — the same pattern used in production ML systems, not manual preprocessing.
- **Input validation server-side**: the Flask API rejects out-of-range or missing fields before they ever reach the model.
- **Shipped, not just modeled**: a working API + a polished React frontend that a non-technical person could actually use.

## Tech stack

- **Model**: scikit-learn (Logistic Regression, with Random Forest / Gradient Boosting as compared baselines)
- **Backend**: Flask + Flask-CORS, serving predictions and model metadata as JSON
- **Frontend**: React (Vite) + Recharts for the model insights charts
- **Data**: synthetic 1,500-student dataset (see note below) shaped like common Kaggle "Campus Placement" datasets

## Project structure

```
placement-predictor/
├── data/
│   ├── generate_data.py       # synthetic dataset generator
│   └── placement_data.csv     # generated dataset
├── backend/
│   ├── train_model.py         # trains, compares, evaluates, saves the model
│   ├── app.py                 # Flask API
│   ├── model.joblib           # trained sklearn pipeline (generated)
│   ├── metrics.json           # evaluation metrics (generated)
│   └── schema.json            # form field definitions (generated)
└── frontend/
    ├── src/
    │   ├── App.jsx
    │   └── components/
    │       ├── PredictForm.jsx
    │       ├── ResultPanel.jsx
    │       └── ModelInsights.jsx
    └── ...
```

## Running it locally

**1. Backend**
```bash
cd backend
pip install flask flask-cors scikit-learn pandas numpy joblib
python app.py
```
Runs at `http://localhost:5000`.

**2. Frontend** (separate terminal)
```bash
cd frontend
npm install
npm run dev
```
Open the printed local URL (typically `http://localhost:5173`).

**3. (Optional) Retrain the model**
```bash
cd data && python generate_data.py
cd ../backend && python train_model.py
```

## A note on the dataset

The dataset is **synthetically generated**, not scraped or downloaded, with engineered relationships between features (CGPA, DSA score, internships, backlogs, etc.) and the placement outcome, plus realistic noise so it's not perfectly separable. This was a deliberate choice over redistributing a Kaggle CSV: it avoids licensing/attribution issues, and it means the data's structure is fully understood and explainable, which matters a lot when you're asked "why does your model make this prediction" in an interview.

For a v2, swapping in a real Kaggle "Campus Placement" dataset (with `pd.read_csv` pointed at the new file) is straightforward — the pipeline doesn't care where the CSV comes from, as long as column names match.

## Ideas for extending this further

- Add SHAP value explanations per-prediction ("your CGPA contributed +12% to your odds")
- Add a salary range regression model (the synthetic data already includes a `salary_lpa` column for this)
- Swap in a real dataset and retrain
- Add authentication + a database to let students save/track multiple profile scenarios
- Deploy: Flask on Render/Railway, frontend on Vercel/Netlify
