# Tabular bias (scikit-learn)

A loan-default model trained on biased records: in its training data, a third of the Southern
applicants who repaid were recorded as defaulting. The model never sees the region, but learns it
from the branch, which PRISM shows. The data also has a column that gives the outcome away.

```bash
python make_data.py      # a few seconds: writes applications.csv and model.joblib
prism
```

| Tab | Set |
|---|---|
| Setup | Project root: this folder · Connector: `connector.py` · Framework: Scikit-Learn |
| Evidence | CSV: `applications.csv` · ID: `applicant_id` · Ground truth: `outcome` |
| Execution | Any output folder |

What you should see:

- **Evidence**: `account_status` predicts the outcome on its own (HIGH): it is recorded after the
  outcome, so a model trained on it would be cheating.
- **Performance**, sliced by `region`: Southern applicants are flagged as defaulting far more
  often, including many who repaid.

Or run every analysis at once: `prism audit <output folder> --slice region sex`.

`connector.py` is a starting point for a pickled scikit-learn model whose inputs are columns of the
CSV: copy it and change the lines marked `CHANGE`.
