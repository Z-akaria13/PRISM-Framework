import joblib

from prism.inspectors.base import Inspector


class SklearnInspector(Inspector):
    """Black box: scikit-learn (and similar) models expose no layers, so only the output is cached."""

    extracts_hidden_layers = False

    def model_fingerprint(self):
        """joblib's hash of the fitted estimator: its parameters and learned values."""
        return "joblib-md5:" + joblib.hash(self.model)
