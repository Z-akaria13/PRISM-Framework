"""core.provenance and the inspectors' model fingerprints (debt P2 #14)."""
import hashlib

import numpy as np
import pytest

import prism
from prism.core.provenance import array_entries, environment, file_digest, weights_digest


def test_the_environment_names_prism_python_and_the_packages_that_shape_the_numbers():
    env = environment()

    assert env["prism"] == prism.__version__
    assert env["python"].count(".") == 2
    assert {"numpy", "pandas", "scikit-learn", "h5py"} <= set(env["packages"])
    assert "tensorflow" not in env["packages"] or env["packages"]["tensorflow"]  # only what is installed


def test_a_files_digest_is_what_sha256sum_prints(tmp_path):
    path = tmp_path / "meta.csv"
    path.write_bytes(b"subject_id,label\n0042,1\n")

    assert file_digest(path) == "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    assert file_digest(tmp_path / "missing.csv") is None and file_digest("") is None


def test_a_weights_digest_ignores_listing_order_but_not_names_values_or_types():
    a, b = np.arange(6.0).reshape(2, 3), np.ones(3, dtype=np.float32)
    digest = weights_digest(array_entries([("a", a), ("b", b)]))

    assert digest == weights_digest(array_entries([("b", b), ("a", a)]))
    assert digest != weights_digest(array_entries([("a", a), ("c", b)]))
    assert digest != weights_digest(array_entries([("a", a + 1e-9), ("b", b)]))
    zeros = [("z", np.zeros(3, np.float32))]  # the same bytes as int32 zeros: only the type differs
    assert weights_digest(array_entries(zeros)) != weights_digest(array_entries([("z", np.zeros(3, np.int32))]))
    assert digest != weights_digest(array_entries([("a", a.reshape(3, 2)), ("b", b)]))


def test_pytorch_fingerprints_weights_and_buffers_whatever_the_mode():
    torch = pytest.importorskip("torch")
    from prism.inspectors.pytorch import PyTorchInspector
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.BatchNorm1d(2)).eval()
    fingerprint = PyTorchInspector(model).model_fingerprint()

    assert PyTorchInspector(model.train()).model_fingerprint() == fingerprint
    with torch.no_grad():
        model[1].running_mean += 1  # a buffer, not a parameter
    assert PyTorchInspector(model).model_fingerprint() != fingerprint
    assert PyTorchInspector(model.to(torch.bfloat16)).model_fingerprint().startswith("sha256:")


def test_scikit_learn_fingerprints_the_fitted_estimator():
    from sklearn.linear_model import LogisticRegression
    from prism.inspectors.sklearn import SklearnInspector
    X, y = np.random.default_rng(0).normal(size=(40, 3)), np.arange(40) % 2

    first = SklearnInspector(LogisticRegression().fit(X, y)).model_fingerprint()

    assert SklearnInspector(LogisticRegression().fit(X, y)).model_fingerprint() == first
    assert SklearnInspector(LogisticRegression().fit(X + 1, y)).model_fingerprint() != first


def test_keras_fingerprints_the_weights_in_keras_order():
    from prism.inspectors.keras import KerasInspector

    class Model:
        def __init__(self, *weights):
            self.weights = list(weights)

        def get_weights(self):
            return self.weights

    a, b = np.zeros((2, 2)), np.ones(2)

    assert KerasInspector(Model(a, b)).model_fingerprint() == KerasInspector(Model(a, b)).model_fingerprint()
    assert KerasInspector(Model(a, b)).model_fingerprint() != KerasInspector(Model(b, a)).model_fingerprint()
