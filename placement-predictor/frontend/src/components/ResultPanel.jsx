const VERDICT_TONE = {
  "Strong Placement Chance": "sage",
  "Moderate Placement Chance": "brass",
  "Needs Improvement": "brass",
  "High Risk — Significant Gaps": "brick",
};

function Gauge({ probability, tone }) {
  const pct = Math.round(probability * 100);
  const radius = 80;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference * (1 - probability);

  return (
    <svg viewBox="0 0 200 200" className={`gauge gauge--${tone}`}>
      <circle cx="100" cy="100" r={radius} className="gauge__track" />
      <circle
        cx="100"
        cy="100"
        r={radius}
        className="gauge__fill"
        strokeDasharray={circumference}
        strokeDashoffset={offset}
        transform="rotate(-90 100 100)"
      />
      <text x="100" y="94" textAnchor="middle" className="gauge__number">
        {pct}%
      </text>
      <text x="100" y="118" textAnchor="middle" className="gauge__caption">
        PLACEMENT ODDS
      </text>
    </svg>
  );
}

export default function ResultPanel({ result, loading }) {
  if (loading) {
    return <p className="muted">Evaluating candidate profile…</p>;
  }

  if (!result) {
    return (
      <div className="result-empty">
        <p>Fill in the candidate profile and submit to see a verdict.</p>
      </div>
    );
  }

  const tone = VERDICT_TONE[result.verdict] || "brass";

  return (
    <div className="result">
      <Gauge probability={result.placement_probability} tone={tone} />
      <p className={`verdict verdict--${tone}`}>{result.verdict}</p>
      <p className="result-note">
        {result.prediction === 1
          ? "Model classifies this profile as likely to be placed."
          : "Model classifies this profile as unlikely to be placed in its current state."}
      </p>
    </div>
  );
}
