"""examples/: each example generates its data, runs end to end, and PRISM finds what was planted."""
import shutil
import subprocess
import sys

import pytest

from _support import EXAMPLES_DIR
from prism.audit import AuditOptions, run_audit
from prism.core.config import AuditConfig
from prism.core.execution import run_extraction
from prism.core.metadata import read_metadata
from prism.core.subject import load_subject

pytestmark = pytest.mark.slow


def run_example(tmp_path, name, csv, framework, id_col, target, group=None, layers=None, **audit):
    """Copy the example, run its make_data.py, extract through its connector, and audit the results."""
    example = tmp_path / name
    shutil.copytree(EXAMPLES_DIR / name, example, ignore=shutil.ignore_patterns("__pycache__"))
    made = subprocess.run([sys.executable, "make_data.py"], cwd=example, capture_output=True, text=True, timeout=300)
    assert made.returncode == 0, made.stderr
    config = AuditConfig(project_root=str(example), script_path=str(example / "connector.py"), framework=framework,
                         csv_path=str(example / csv), id_col=id_col, target_col=target, group_col=group,
                         layers=layers or {}, output_dir=str(tmp_path / "out"), batch_size=64)
    (tmp_path / "out").mkdir()
    subject = load_subject(config.project_root, config.script_path, config.framework)
    logs = []
    assert run_extraction(config, subject, read_metadata(config.csv_path, id_col), lambda p: None, logs.append), logs
    assert not any("[WARNING]" in line for line in logs), logs  # nothing left out, no class-order guess
    return run_audit(str(tmp_path / "out"), AuditOptions(bootstrap=300, **audit), log=lambda message: None)


def flagged(result, column):
    """{gap name: (favoured group, other group)} for the gaps flagged in one slice column."""
    return {name: (gap["high"], gap["low"]) for name, gap in result["performance"]["slices"][column]["gaps"].items()
            if gap["flagged"]}


def test_the_image_example_shows_its_shortcut_and_its_sex_gap(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("PIL")

    result = run_example(tmp_path, "image_shortcut", "metadata.csv", "PyTorch", "scan_id", "diagnosis",
                         group="patient_id", layers={"conv1": "Mean Pool", "conv2": "Mean Pool", "fc1": "Mean Pool"},
                         slices=["site", "sex"], concepts=["site"])

    assert result["rows"] == 400 and result["subjects"] == 200
    site, sex = flagged(result, "site"), flagged(result, "sex")
    assert site.get("equal_opportunity") == ("B", "A")  # disease missed more often at site A
    assert site.get("demographic_parity") == ("B", "A")  # and called more often at site B
    assert sex.get("equal_opportunity") == ("M", "F")    # disease missed more often in women
    encoded = {row["layer"] for row in result["probes"]["rows"] if row["concept"] == "site" and row["significant"]}
    assert {"conv2", "fc1"} <= encoded


@pytest.mark.filterwarnings("error::UserWarning")  # e.g. scikit-learn's "X does not have valid feature names"
def test_the_tabular_example_shows_its_label_bias_and_its_leak(tmp_path):
    result = run_example(tmp_path, "tabular_bias", "applications.csv", "Scikit-Learn", "applicant_id", "outcome",
                         slices=["region", "sex"])

    assert result["rows"] == 1000
    region = flagged(result, "region")
    assert region.get("demographic_parity") == ("South", "North")  # flagged as defaulting far more often
    assert region.get("predictive_equality") == ("South", "North")  # including many who repaid
    assert not flagged(result, "sex")
    leaks = {row["field"]: row["risk"] for row in result["screening"]}
    assert leaks["account_status"] in ("HIGH", "EXTREME")
    assert leaks["region"] not in ("HIGH", "EXTREME")  # the outcome itself is the same in both regions


def test_the_connectors_need_only_prism_and_their_project(tmp_path):
    for name in ("image_shortcut", "tabular_bias"):
        source = (EXAMPLES_DIR / name / "connector.py").read_text(encoding="utf-8")
        assert "from prism import PrismConnector" in source and "CHANGE" in source
        assert "core.connectorTemplate" not in source
