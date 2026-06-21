import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, Legend,
} from "recharts";

const FEATURE_LABELS = {
  cgpa: "CGPA",
  dsa_score: "DSA Score",
  internships: "Internships",
  backlogs: "Backlogs",
  communication_score: "Communication",
  projects: "Projects",
  certifications: "Certifications",
  ssc_percentage: "SSC %",
  hsc_percentage: "HSC %",
  extra_curricular: "Extra-curricular",
};

function prettyFeature(name) {
  if (FEATURE_LABELS[name]) return FEATURE_LABELS[name];
  if (name.startsWith("branch_")) return `Branch: ${name.replace("branch_", "")}`;
  if (name.startsWith("specialization_")) return `Spec: ${name.replace("specialization_", "")}`;
  return name;
}

export default function ModelInsights({ metrics }) {
  const topFeatures = metrics.feature_importance.slice(0, 8).map((f) => ({
    name: prettyFeature(f.feature),
    importance: Number(f.importance.toFixed(3)),
  }));

  const roc = metrics.roc_curve.fpr.map((fpr, i) => ({
    fpr: Number(fpr.toFixed(2)),
    tpr: Number(metrics.roc_curve.tpr[i].toFixed(2)),
  }));
  const diagonal = roc.map((p) => ({ fpr: p.fpr, baseline: p.fpr }));
  const rocMerged = roc.map((p, i) => ({ ...p, baseline: diagonal[i].baseline }));

  const cm = metrics.confusion_matrix;

  return (
    <div className="insights">
      <div className="insights__metrics">
        <Metric label="Accuracy" value={metrics.accuracy} />
        <Metric label="Precision" value={metrics.precision} />
        <Metric label="Recall" value={metrics.recall} />
        <Metric label="F1 Score" value={metrics.f1_score} />
        <Metric label="ROC-AUC" value={metrics.roc_auc} />
      </div>

      <div className="insights__grid">
        <div className="insight-card">
          <h3>Feature Importance</h3>
          <p className="insight-card__sub">What the model weighs most heavily</p>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={topFeatures} layout="vertical" margin={{ left: 8, right: 16 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2A333C" horizontal={false} />
              <XAxis type="number" stroke="#B8B2A4" fontSize={11} />
              <YAxis
                type="category"
                dataKey="name"
                stroke="#B8B2A4"
                fontSize={11}
                width={110}
              />
              <Tooltip
                contentStyle={{ background: "#161D24", border: "1px solid #2A333C", fontSize: 12 }}
                labelStyle={{ color: "#F5F1E8" }}
              />
              <Bar dataKey="importance" fill="#C9A876" radius={[0, 3, 3, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="insight-card">
          <h3>ROC Curve</h3>
          <p className="insight-card__sub">True positive rate vs. false positive rate</p>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={rocMerged} margin={{ left: 8, right: 16 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2A333C" />
              <XAxis dataKey="fpr" stroke="#B8B2A4" fontSize={11} label={{ value: "FPR", position: "insideBottom", offset: -4, fill: "#B8B2A4", fontSize: 11 }} />
              <YAxis stroke="#B8B2A4" fontSize={11} label={{ value: "TPR", angle: -90, position: "insideLeft", fill: "#B8B2A4", fontSize: 11 }} />
              <Tooltip
                contentStyle={{ background: "#161D24", border: "1px solid #2A333C", fontSize: 12 }}
                labelStyle={{ color: "#F5F1E8" }}
              />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Line type="monotone" dataKey="tpr" stroke="#5C9484" strokeWidth={2} dot={false} name="Model" />
              <Line type="monotone" dataKey="baseline" stroke="#4A302A" strokeWidth={1.5} strokeDasharray="4 4" dot={false} name="Random guess" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="insight-card">
          <h3>Confusion Matrix</h3>
          <p className="insight-card__sub">Held-out test set (n = {cm.tn + cm.fp + cm.fn + cm.tp})</p>
          <div className="cm-grid">
            <div className="cm-cell cm-cell--tn">
              <span className="cm-cell__value">{cm.tn}</span>
              <span className="cm-cell__label">True Negative</span>
            </div>
            <div className="cm-cell cm-cell--fp">
              <span className="cm-cell__value">{cm.fp}</span>
              <span className="cm-cell__label">False Positive</span>
            </div>
            <div className="cm-cell cm-cell--fn">
              <span className="cm-cell__value">{cm.fn}</span>
              <span className="cm-cell__label">False Negative</span>
            </div>
            <div className="cm-cell cm-cell--tp">
              <span className="cm-cell__value">{cm.tp}</span>
              <span className="cm-cell__label">True Positive</span>
            </div>
          </div>
        </div>

        <div className="insight-card">
          <h3>Model Comparison</h3>
          <p className="insight-card__sub">5-fold cross-validation, ROC-AUC</p>
          <table className="cv-table">
            <thead>
              <tr><th>Model</th><th>Mean AUC</th><th>Std</th></tr>
            </thead>
            <tbody>
              {Object.entries(metrics.cv_comparison).map(([name, v]) => (
                <tr key={name} className={name === metrics.model_name ? "cv-table__winner" : ""}>
                  <td>{name}{name === metrics.model_name && " ★"}</td>
                  <td>{v.mean_auc}</td>
                  <td>±{v.std_auc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span className="metric__value">{(value * 100).toFixed(1)}</span>
      <span className="metric__label">{label}</span>
    </div>
  );
}
