"""
Generates a synthetic but realistic campus placement dataset.
Mirrors the structure/feel of common Kaggle "Campus Placement" datasets,
with added features relevant to CS/MCA students (DSA score, projects, certifications).

Why synthetic: lets us control realistic relationships between features and
the placement outcome (so the model has real signal to learn), and avoids
licensing/attribution issues with redistributing a Kaggle CSV.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 1500

branches = ["CSE", "IT", "MCA", "ECE", "Mech", "Civil"]
branch_weights = [0.28, 0.18, 0.18, 0.16, 0.12, 0.08]

specializations = ["AI/ML", "Web Dev", "Data Science", "Cloud/DevOps", "Core/Other"]

df = pd.DataFrame({
    "cgpa": np.round(np.random.normal(7.3, 0.9, N).clip(5.0, 10.0), 2),
    "ssc_percentage": np.round(np.random.normal(78, 9, N).clip(45, 99), 2),
    "hsc_percentage": np.round(np.random.normal(75, 10, N).clip(40, 99), 2),
    "branch": np.random.choice(branches, N, p=branch_weights),
    "specialization": np.random.choice(specializations, N),
    "internships": np.random.poisson(1.1, N).clip(0, 5),
    "projects": np.random.poisson(2.2, N).clip(0, 8),
    "certifications": np.random.poisson(1.5, N).clip(0, 6),
    "dsa_score": np.round(np.random.normal(60, 20, N).clip(0, 100), 1),     # coding test / LeetCode-style score
    "communication_score": np.round(np.random.normal(65, 15, N).clip(0, 100), 1),  # soft-skill / interview score
    "backlogs": np.random.choice([0, 0, 0, 1, 2], N, p=[0.55, 0.2, 0.15, 0.07, 0.03]),
    "extra_curricular": np.random.choice([0, 1], N, p=[0.6, 0.4]),  # club/hackathon participation
})

# Build a latent "placement score" from a weighted combination of features,
# so the outcome has genuine, learnable structure (not pure noise).
score = (
    df["cgpa"] * 9.5
    + df["dsa_score"] * 0.35
    + df["communication_score"] * 0.25
    + df["internships"] * 6
    + df["projects"] * 3
    + df["certifications"] * 2
    + df["extra_curricular"] * 4
    - df["backlogs"] * 12
    + (df["ssc_percentage"] + df["hsc_percentage"]) * 0.08
)

# Add a modest amount of noise so it's not perfectly separable (placements involve
# some luck/fit), but keep signal dominant so the model has real patterns to learn.
score += np.random.normal(0, 9, N)

# Convert score to probability via logistic function, then sample binary outcome.
# Dividing by a smaller constant sharpens the sigmoid so the engineered signal
# (cgpa, dsa_score, internships, etc.) actually drives separability instead of
# being washed out into near-random 50/50 outcomes.
prob = 1 / (1 + np.exp(-(score - score.mean()) / 8))
df["placed"] = np.random.binomial(1, prob)

# Add a salary (LPA) for placed students only — useful bonus target / feature
base_salary = 3.5 + (df["cgpa"] - 6) * 0.9 + df["dsa_score"] * 0.03 + df["internships"] * 0.4
df["salary_lpa"] = np.where(
    df["placed"] == 1,
    np.round((base_salary + np.random.normal(0, 1.2, N)).clip(2.5, 30), 2),
    np.nan,
)

df.insert(0, "student_id", range(1, N + 1))

df.to_csv("/home/claude/placement-predictor/data/placement_data.csv", index=False)
print(f"Generated {len(df)} rows")
print(f"Placement rate: {df['placed'].mean():.1%}")
print(df.head())
