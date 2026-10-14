"""Opt-in scale test: `pytest -m scale`. 100,000 rows through extraction and a probe, to catch
anything that quietly grows faster than the data (e.g. an accidentally quadratic step)."""
import time

import numpy as np
import pandas as pd
import pytest

from _support import ValueConnector, run_engine
from prism.analysis import linear_probe
from prism.core.cache import AuditCache

pytestmark = pytest.mark.scale

N, D = 100_000, 256


class LayerFromTable:
    """Duck-typed inspector serving one precomputed 256-feature layer for whichever rows were inferred."""

    extracts_hidden_layers = True

    def __init__(self, table):
        self.table, self.pending = table, {}

    def attach_hooks(self, names):
        pass

    def retrieve_features(self, expected_rows=None):
        return dict(self.pending)

    def clear_buffer(self):
        self.pending = {}

    def teardown(self):
        pass


class TableConnector(ValueConnector):
    def __init__(self, inspector):
        super().__init__()
        self.inspector = inspector

    def load_item(self, row_id, meta):
        return int(meta["value"])

    def run_inference(self, model, batch):
        self.inspector.pending = {"layer": self.inspector.table[batch]}
        return self.inspector.table[batch, 0]


def test_a_hundred_thousand_rows_extract_and_probe(tmp_path):
    rng = np.random.default_rng(0)
    patient = np.repeat(np.arange(N // 4), 4)
    sex = rng.integers(0, 2, N // 4)[patient]
    table = rng.normal(size=(N, D)).astype(np.float32)
    table[:, :8] += 0.3 * sex[:, None]  # a clear signal: the point is scale, not sensitivity
    metadata = pd.DataFrame({"subject_id": [f"S{i:06d}" for i in range(N)], "value": np.arange(N),
                             "label": sex, "patient": patient})
    inspector = LayerFromTable(table)

    start = time.perf_counter()
    run = run_engine(tmp_path / "out", metadata, TableConnector(inspector), inspector,
                     layers=["layer"], batch_size=2000, group_col="patient")
    extraction = time.perf_counter() - start

    assert run.ok, run.logs[-3:]
    cache = AuditCache.open(str(run.out_dir))
    assert len(cache.metadata) == N and cache.layer_names == ["layer"]

    start = time.perf_counter()
    with cache.open_layer("layer") as layer:
        result = linear_probe(layer, cache.metadata["label"], groups=cache.metadata["patient"], n_controls=50)
    probing = time.perf_counter() - start

    assert result["auc_mean"] > 0.6 and result["control"]["p_value"] == pytest.approx(1 / 51)
    print(f"\n100k rows: extraction {extraction:.1f} s, grouped LDA probe with 50 shuffles {probing:.1f} s")
    assert extraction < 600 and probing < 600  # generous: this guards against blow-ups, not small slowdowns


def test_fairness_intervals_for_a_hundred_thousand_rows():
    from prism.analysis import fairness_audit
    rng = np.random.default_rng(0)
    patient = np.repeat(np.arange(N // 4), 4)
    site = rng.integers(0, 5, N // 4)[patient]
    y_true = rng.integers(0, 3, N)
    y_pred = np.where(rng.random(N) < 0.8, y_true, rng.integers(0, 3, N))

    timings = {}
    for unit, groups in [("subjects", patient), ("scans", None)]:
        start = time.perf_counter()
        result = fairness_audit(y_true, y_pred, site, groups=groups)
        timings[unit] = time.perf_counter() - start
        assert result["bootstrap"]["replicates"] == 1000 and len(result["groups"]) == 5

    print(f"\n100k rows, 3 classes, 5 sites, 1000 resamples: {timings['subjects']:.1f} s by subject, "
          f"{timings['scans']:.1f} s by scan")
    assert max(timings.values()) < 120
