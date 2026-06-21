import { useState } from "react";

const FIELD_GROUPS = [
  {
    title: "Academic Record",
    fields: ["cgpa", "ssc_percentage", "hsc_percentage", "backlogs"],
  },
  {
    title: "Experience",
    fields: ["internships", "projects", "certifications", "extra_curricular"],
  },
  {
    title: "Assessment Scores",
    fields: ["dsa_score", "communication_score"],
  },
];

export default function PredictForm({ schema, onSubmit, loading }) {
  const defaults = Object.fromEntries(
    Object.entries(schema.numeric_features).map(([key, cfg]) => [key, cfg.default])
  );
  defaults.branch = schema.categorical_features.branch[0];
  defaults.specialization = schema.categorical_features.specialization[0];

  const [values, setValues] = useState(defaults);

  function setField(key, val) {
    setValues((prev) => ({ ...prev, [key]: val }));
  }

  function handleSubmit(e) {
    e.preventDefault();
    onSubmit(values);
  }

  return (
    <form className="predict-form" onSubmit={handleSubmit}>
      {FIELD_GROUPS.map((group) => (
        <fieldset key={group.title} className="form-group">
          <legend>{group.title}</legend>
          {group.fields.map((key) => {
            const cfg = schema.numeric_features[key];
            const isBinary = cfg.max === 1 && cfg.min === 0 && cfg.step === 1;
            if (isBinary) {
              return (
                <label key={key} className="field field--toggle">
                  <span className="field__label">{cfg.label}</span>
                  <input
                    type="checkbox"
                    checked={values[key] === 1}
                    onChange={(e) => setField(key, e.target.checked ? 1 : 0)}
                  />
                </label>
              );
            }
            return (
              <label key={key} className="field">
                <span className="field__label">
                  {cfg.label}
                  <span className="field__value">{values[key]}</span>
                </span>
                <input
                  type="range"
                  min={cfg.min}
                  max={cfg.max}
                  step={cfg.step}
                  value={values[key]}
                  onChange={(e) => setField(key, parseFloat(e.target.value))}
                />
              </label>
            );
          })}
        </fieldset>
      ))}

      <fieldset className="form-group">
        <legend>Background</legend>
        <label className="field">
          <span className="field__label">Branch</span>
          <select value={values.branch} onChange={(e) => setField("branch", e.target.value)}>
            {schema.categorical_features.branch.map((b) => (
              <option key={b} value={b}>{b}</option>
            ))}
          </select>
        </label>
        <label className="field">
          <span className="field__label">Specialization</span>
          <select
            value={values.specialization}
            onChange={(e) => setField("specialization", e.target.value)}
          >
            {schema.categorical_features.specialization.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </label>
      </fieldset>

      <button type="submit" className="submit-btn" disabled={loading}>
        {loading ? "Evaluating…" : "Predict Placement Odds"}
      </button>
    </form>
  );
}
