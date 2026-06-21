import { useEffect, useState } from "react";
import "./App.css";
import PredictForm from "./components/PredictForm";
import ResultPanel from "./components/ResultPanel";
import ModelInsights from "./components/ModelInsights";

const API_BASE = "http://localhost:5000/api";

export default function App() {
  const [schema, setSchema] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [apiDown, setApiDown] = useState(false);

  useEffect(() => {
    Promise.all([
      fetch(`${API_BASE}/schema`).then((r) => r.json()),
      fetch(`${API_BASE}/metrics`).then((r) => r.json()),
    ])
      .then(([schemaData, metricsData]) => {
        setSchema(schemaData);
        setMetrics(metricsData);
      })
      .catch(() => setApiDown(true));
  }, []);

  async function handlePredict(formData) {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });
      const data = await res.json();
      if (!res.ok) {
        setError(data.error || "Prediction failed");
        setResult(null);
      } else {
        setResult(data);
      }
    } catch (e) {
      setApiDown(true);
    } finally {
      setLoading(false);
    }
  }

  if (apiDown) {
    return (
      <div className="api-down">
        <p className="api-down__eyebrow">CONNECTION FAILED</p>
        <h1>The prediction service isn't responding.</h1>
        <p>
          Start the Flask backend first: <code>cd backend && python app.py</code>
          <br />
          It should be running at <code>http://localhost:5000</code>.
        </p>
      </div>
    );
  }

  return (
    <div className="page">
      <header className="masthead">
        <div className="masthead__seal">PP</div>
        <div className="masthead__text">
          <p className="masthead__eyebrow">CAREER OUTCOME DOSSIER</p>
          <h1 className="masthead__title">Placement Predictor</h1>
        </div>
        {metrics && (
          <div className="masthead__stat">
            <span className="masthead__stat-value">{(metrics.roc_auc * 100).toFixed(0)}</span>
            <span className="masthead__stat-label">ROC&#8209;AUC&nbsp;×100</span>
          </div>
        )}
      </header>

      <main className="layout">
        <section className="panel panel--form">
          <h2 className="panel__heading">I. Candidate Profile</h2>
          {schema ? (
            <PredictForm schema={schema} onSubmit={handlePredict} loading={loading} />
          ) : (
            <p className="muted">Loading form…</p>
          )}
          {error && <p className="form-error">⚠ {error}</p>}
        </section>

        <section className="panel panel--result">
          <h2 className="panel__heading">II. Verdict</h2>
          <ResultPanel result={result} loading={loading} />
        </section>
      </main>

      {metrics && (
        <section className="panel panel--insights">
          <h2 className="panel__heading">III. Model Insights</h2>
          <ModelInsights metrics={metrics} />
        </section>
      )}

      <footer className="footer">
        <p>
          Trained on a synthetic 1,500-student dataset · Logistic Regression ·
          built as a portfolio project
        </p>
      </footer>
    </div>
  );
}
