"""headless.py: the cluster worker, run as a real subprocess from a JSON job file."""
import os
import subprocess
import sys

import h5py
import numpy as np
import pandas as pd
import pytest

from _support import (BLACK_BOX_CONNECTOR, CSV_NAME, H5_NAME, HEADLESS, TORCH_CONNECTOR,
                      make_project, run_headless, write_job)

pytestmark = pytest.mark.slow


def job(tmp_path, connector_source, framework, layers):
    script, csv = make_project(tmp_path / "project", connector_source, n=6)
    out = tmp_path / "out"
    out.mkdir()
    config = write_job(tmp_path / "job.json", script=script, csv=csv, project_root=script.parent,
                       output_dir=out, framework=framework, layers=layers)
    return config, out


def test_black_box_job_produces_a_cache(tmp_path):
    config, out = job(tmp_path, BLACK_BOX_CONNECTOR, "Scikit-Learn", layers={})

    result = run_headless(config, cwd=tmp_path)

    assert result.returncode == 0, result.stderr
    assert "[SUCCESS] Extracted 6/6" in result.stdout
    with h5py.File(out / H5_NAME, "r") as f:
        np.testing.assert_allclose(f["__model_output__"][:], np.arange(6) / 10)
    assert len(pd.read_csv(out / CSV_NAME)) == 6


@pytest.mark.parametrize("layers", [
    {"conv": "Max Pool", "fc": "Mean Pool"},
    ["conv [Max Pool]", "fc"],  # job files written before AuditConfig
], ids=["current-format", "legacy-labels"])
def test_pytorch_job_extracts_selected_layers_with_their_pooling(tmp_path, layers):
    config, out = job(tmp_path, TORCH_CONNECTOR, "PyTorch", layers=layers)

    result = run_headless(config, cwd=tmp_path)

    assert result.returncode == 0, result.stderr
    assert "Running on device: cpu" in result.stdout
    with h5py.File(out / H5_NAME, "r") as f:
        assert f["conv"].shape == (6, 3)
        assert f["fc"].shape == (6, 2)
        np.testing.assert_allclose(f["fc"][:], f["__model_output__"][:], rtol=1e-6)


def test_unknown_framework_fails_the_job(tmp_path):
    config, _ = job(tmp_path, BLACK_BOX_CONNECTOR, "JAX", layers={})

    result = run_headless(config, cwd=tmp_path)

    assert result.returncode != 0
    assert "Framework 'JAX' not found" in result.stderr


def test_failed_extraction_exits_non_zero(tmp_path):
    config, _ = job(tmp_path, TORCH_CONNECTOR, "PyTorch", layers={})  # white box, nothing selected

    result = run_headless(config, cwd=tmp_path)

    assert result.returncode == 1
    assert "[CRITICAL ERROR] No layers chosen" in result.stdout


def test_job_reports_progress_for_the_gui_monitor(tmp_path):
    config, _ = job(tmp_path, BLACK_BOX_CONNECTOR, "Scikit-Learn", layers={})

    result = run_headless(config, cwd=tmp_path)

    progress = [line for line in result.stdout.splitlines() if line.startswith("[PROGRESS]")]
    assert progress == ["[PROGRESS] 50.0", "[PROGRESS] 100.0"]  # 6 rows, batches of 4


LIVE_LOG_CONNECTOR = '''
import os
import numpy as np
from core.connectorTemplate import WhydunnitConnector

class LiveLogProbe(WhydunnitConnector):
    """During batch 2, record whether batch 1's progress line has already reached the log file."""

    def load_model(self):
        return None

    def load_item(self, row_id, meta):
        return float(meta["value"])

    def run_inference(self, model, batch):
        if 4.0 in batch:
            with open(os.environ["PRISM_TEST_LOG"]) as log:
                seen = "[PROGRESS] 50.0" in log.read()
            with open(os.environ["PRISM_TEST_MARKER"], "w") as marker:
                marker.write("live" if seen else "buffered")
        return np.asarray(batch)

    def render_item_for_gallery(self, item):
        return {"type": "text", "payload": str(item)}
'''


def test_output_reaches_a_redirected_log_while_the_job_runs(tmp_path):
    """The cluster script sends stdout to a file the GUI tails; it must not sit in Python's buffer."""
    config, _ = job(tmp_path, LIVE_LOG_CONNECTOR, "Scikit-Learn", layers={})
    log, marker = tmp_path / "job.log", tmp_path / "marker.txt"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONUNBUFFERED"}
    env.update(PRISM_TEST_LOG=str(log), PRISM_TEST_MARKER=str(marker))

    with open(log, "w") as fh:  # like `python headless.py ... > log 2>&1`, deliberately without -u
        subprocess.run([sys.executable, str(HEADLESS), "--config", str(config)], cwd=str(tmp_path),
                       stdout=fh, stderr=subprocess.STDOUT, env=env, timeout=300, check=True)

    assert marker.read_text() == "live"


def test_a_job_refuses_to_replace_existing_results(tmp_path):
    config, _ = job(tmp_path, BLACK_BOX_CONNECTOR, "Scikit-Learn", layers={})
    assert run_headless(config, cwd=tmp_path).returncode == 0

    second = run_headless(config, cwd=tmp_path)

    assert second.returncode == 1
    assert "Results already exist" in second.stdout


def test_a_job_whose_rows_all_fail_exits_non_zero_and_saves_nothing(tmp_path):
    unreadable = BLACK_BOX_CONNECTOR.replace('return float(meta["value"])',
                                             'raise FileNotFoundError(f"scans/{row_id}.nii")')
    assert unreadable != BLACK_BOX_CONNECTOR
    config, out = job(tmp_path, unreadable, "Scikit-Learn", layers={})

    result = run_headless(config, cwd=tmp_path)

    assert result.returncode == 1
    assert "[CRITICAL ERROR] Stopped: more than 5% of the 6 rows failed" in result.stdout
    assert "FileNotFoundError: scans/S000.nii" in result.stdout
    assert "[SUCCESS]" not in result.stdout
    assert [p.name for p in out.iterdir()] == []


FILE_CONNECTOR = """
import os
import numpy as np
from prism import PrismConnector

class FileConnector(PrismConnector):
    def load_model(self):
        return None

    def load_item(self, row_id, meta):
        with open(os.path.join(self.project_root, "items", f"{row_id}.txt")) as f:
            return float(f.read())

    def run_inference(self, model, batch):
        return np.asarray(batch)
"""


def test_ids_with_leading_zeros_reach_the_connector_intact(tmp_path):
    # debt P1 #25: read with pandas' default types, 0042 became "42" and loaded the decoy 42.txt
    script, _ = make_project(tmp_path / "project", FILE_CONNECTOR)
    project = script.parent
    csv = project / "zeros.csv"
    csv.write_text("subject_id,label\n0042,0\n0043,1\n0100,0\n")
    (project / "items").mkdir()
    for name, value in [("0042", 1), ("0043", 2), ("0100", 3), ("42", 99), ("43", 99), ("100", 99)]:
        (project / "items" / f"{name}.txt").write_text(str(value))
    out = tmp_path / "out"
    out.mkdir()
    config = write_job(tmp_path / "job.json", script=script, csv=csv, project_root=project, output_dir=out,
                       framework="Scikit-Learn", layers={})

    result = run_headless(config, cwd=tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    with h5py.File(out / H5_NAME, "r") as f:
        np.testing.assert_array_equal(f["__model_output__"][:], [1, 2, 3])
        assert list(f["__ids__"].asstr()[:]) == ["0042", "0043", "0100"]
    assert list(pd.read_csv(out / CSV_NAME, dtype=str)["subject_id"]) == ["0042", "0043", "0100"]


def test_a_job_records_what_produced_its_results(tmp_path):
    import hashlib
    import prism
    from prism.core.cache import AuditCache
    config, out = job(tmp_path, BLACK_BOX_CONNECTOR, "Scikit-Learn", layers={})

    result = run_headless(config, cwd=tmp_path)

    assert result.returncode == 0, result.stderr
    p = AuditCache.open(str(out)).provenance
    project = tmp_path / "project"
    for key, path in (("metadata_csv", project / "meta.csv"), ("connector", project / "connector.py")):
        assert p["inputs"][key] == {"path": str(path), "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()}
    assert p["inputs"]["model"].startswith("joblib-md5:")  # the scikit-learn plugin's fingerprint
    assert f"Model fingerprint: {p['inputs']['model']}" in result.stdout
    assert p["prism"] == prism.__version__ and p["started"] <= p["finished"]
