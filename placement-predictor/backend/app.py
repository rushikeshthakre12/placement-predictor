"""
Flask API for the Placement Predictor.

Endpoints:
  GET  /api/health    -> health check
  GET  /api/schema     -> feature definitions for building the form (ranges, dropdown options)
  GET  /api/metrics    -> model evaluation metrics (for the "Model Insights" page)
  POST /api/predict    -> takes a student profile, returns placement probability + verdict
"""

import json
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
CORS(app)  # allow the React dev server to call this API during local development

# Load model + metadata once at startup (not per-request) for performance
model = joblib.load(os.path.join(BASE_DIR, "model.joblib"))

with open(os.path.join(BASE_DIR, "schema.json")) as f:
    SCHEMA = json.load(f)

with open(os.path.join(BASE_DIR, "metrics.json")) as f:
    METRICS = json.load(f)

NUMERIC_FIELDS = list(SCHEMA["numeric_features"].keys())
CATEGORICAL_FIELDS = list(SCHEMA["categorical_features"].keys())
REQUIRED_FIELDS = NUMERIC_FIELDS + CATEGORICAL_FIELDS


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/api/schema")
def get_schema():
    return jsonify(SCHEMA)


@app.get("/api/metrics")
def get_metrics():
    return jsonify(METRICS)


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Request body must be JSON"}), 400

    missing = [f for f in REQUIRED_FIELDS if f not in payload]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    # Validate numeric fields are within sane ranges (basic input validation —
    # protects the model from nonsensical inputs like CGPA of 50)
    for field in NUMERIC_FIELDS:
        bounds = SCHEMA["numeric_features"][field]
        try:
            value = float(payload[field])
        except (TypeError, ValueError):
            return jsonify({"error": f"Field '{field}' must be a number"}), 400
        if not (bounds["min"] <= value <= bounds["max"]):
            return jsonify({
                "error": f"Field '{field}' must be between {bounds['min']} and {bounds['max']}"
            }), 400

    for field in CATEGORICAL_FIELDS:
        if payload[field] not in SCHEMA["categorical_features"][field]:
            return jsonify({
                "error": f"Field '{field}' must be one of {SCHEMA['categorical_features'][field]}"
            }), 400

    row = {f: payload[f] for f in REQUIRED_FIELDS}
    X = pd.DataFrame([row])

    proba = float(model.predict_proba(X)[0, 1])
    prediction = int(proba >= 0.5)

    # Simple human-readable verdict bands for the UI
    if proba >= 0.75:
        verdict = "Strong Placement Chance"
    elif proba >= 0.5:
        verdict = "Moderate Placement Chance"
    elif proba >= 0.3:
        verdict = "Needs Improvement"
    else:
        verdict = "High Risk — Significant Gaps"

    return jsonify({
        "placement_probability": round(proba, 4),
        "prediction": prediction,
        "verdict": verdict,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
