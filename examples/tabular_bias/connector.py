"""
PRISM connector for a pickled scikit-learn model whose inputs are columns of the CSV.
Copy it into your project and change the marked lines.
"""
import os

import joblib
import pandas as pd

from prism import PrismConnector

FEATURES = ["income", "credit_score", "debt_ratio", "branch"]  # CHANGE: the columns your model was fitted on


class TabularConnector(PrismConnector):
    # The target column's labels as [negative, positive]; run_inference returns P(positive).
    class_names = ["repaid", "default"]  # CHANGE

    def load_model(self):
        return joblib.load(os.path.join(self.project_root, "model.joblib"))  # CHANGE: file name

    def load_item(self, row_id, metadata_row):
        return {name: metadata_row[name] for name in FEATURES}

    def run_inference(self, model, batch_data):
        features = pd.DataFrame(batch_data, columns=FEATURES)  # the same column names the model was fitted with
        positive = list(model.classes_).index(self.class_names[1])  # scikit-learn sorts the labels it saw
        return model.predict_proba(features)[:, positive]

    def render_item_for_gallery(self, raw_data_item):
        return {"type": "table", "payload": raw_data_item}
