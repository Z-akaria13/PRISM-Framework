"""
Generate this example's loan applications and train its model (a few seconds):

    python make_data.py

Writes, next to this file:
    applications.csv   1,000 applications to audit: applicant_id, region, sex, age, income,
                       credit_score, debt_ratio, branch, account_status, outcome
    model.joblib       the trained scikit-learn model (features: income, credit_score,
                       debt_ratio, branch)

What is planted, for PRISM to find:
    - Whether an applicant repays depends only on income, credit score and debt ratio, the same
      in both regions. But in the training records, a third of the Southern applicants who
      repaid were recorded as defaulting. The model never sees the region, yet learns it from
      the branch (branches 7-9 are in the South), so it flags Southern applicants far more often.
    - account_status is recorded after the outcome (collections close defaulted accounts), so it
      gives the outcome away. It isn't a model input; the Evidence tab's screen flags it.
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
FEATURES = ["income", "credit_score", "debt_ratio", "branch"]
MISRECORDED_SOUTH = 1 / 3  # share of Southern repayments recorded as defaults in the training data
SEED = 0


def applications(rng, n, start=1):
    region = rng.choice(["North", "South"], n, p=[0.6, 0.4])
    data = pd.DataFrame({
        "applicant_id": [f"A{i:05d}" for i in range(start, start + n)],
        "region": region,
        "sex": rng.choice(["F", "M"], n),
        "age": rng.integers(21, 75, n),
        "income": np.round(rng.lognormal(np.log(45), 0.4, n), 1),  # thousands a year
        "credit_score": np.clip(rng.normal(650, 70, n), 300, 850).round().astype(int),
        "debt_ratio": np.round(rng.beta(2, 4, n), 3),
        "branch": np.where(region == "North", rng.integers(1, 7, n), rng.integers(7, 10, n)),
    })
    logit = (-1.0 - 0.012 * (data["credit_score"] - 650) + 4.0 * (data["debt_ratio"] - 0.33)
             - 0.02 * (data["income"] - 45))
    data["default"] = rng.random(n) < 1 / (1 + np.exp(-logit))
    return data


def main():
    rng = np.random.default_rng(SEED)

    training = applications(rng, 4000, start=50001)
    recorded = training["default"] | ((training["region"] == "South") & (rng.random(len(training)) < MISRECORDED_SOUTH))
    model = GradientBoostingClassifier(random_state=SEED)
    model.fit(training[FEATURES], np.where(recorded, "default", "repaid"))
    joblib.dump(model, os.path.join(HERE, "model.joblib"))

    audited = applications(rng, 1000)
    audited["account_status"] = np.where(audited["default"] & (rng.random(len(audited)) < 0.85),
                                         "closed_by_collections", "open")
    audited["outcome"] = np.where(audited["default"], "default", "repaid")
    audited.drop(columns="default").to_csv(os.path.join(HERE, "applications.csv"), index=False)
    print(f"Wrote applications.csv ({len(audited)} rows) and model.joblib to {HERE}")


if __name__ == "__main__":
    main()
