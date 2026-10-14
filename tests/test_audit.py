"""prism/audit.py: the whole-cache audit, its false-discovery-rate correction, the report and the CLI."""
import json
import math

import numpy as np
import pandas as pd
import pytest

from _support import ValueConnector, run_engine, write_legacy_cache
from prism.analysis import benjamini_hochberg, controls_for_fdr
from prism.audit import REPORT_NAME, AuditError, AuditOptions, main, render_report, run_audit, to_json


# ---------------------------------------------------------------------------
# Benjamini-Hochberg
# ---------------------------------------------------------------------------

def test_benjamini_hochberg_hand_checked():
    # sorted 0.005, 0.01, 0.03, 0.04 -> x4/1, x4/2, x4/3, x4/4 = 0.02, 0.02, 0.04, 0.04 (already monotone)
    assert benjamini_hochberg([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.02, 0.04, 0.04, 0.02])


def test_benjamini_hochberg_takes_the_running_minimum_from_the_top():
    # p*m/rank: 0.03, 0.01875, 0.025; the first is lowered to the smaller value above it
    assert benjamini_hochberg([0.01, 0.0125, 0.025]) == pytest.approx([0.01875, 0.01875, 0.025])


def test_benjamini_hochberg_matches_scipy():
    scipy_stats = pytest.importorskip("scipy.stats")
    if not hasattr(scipy_stats, "false_discovery_control"):
        pytest.skip("scipy < 1.11")
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = rng.random(rng.integers(1, 30)) ** 3
        assert benjamini_hochberg(p) == pytest.approx(scipy_stats.false_discovery_control(p, method="bh"))


def test_untested_entries_stay_nan_and_do_not_count():
    q = benjamini_hochberg([0.01, float("nan"), 0.04])

    assert math.isnan(q[1])
    assert q[[0, 2]] == pytest.approx([0.02, 0.04])  # m = 2, not 3
    assert benjamini_hochberg([]).shape == (0,)


@pytest.mark.parametrize("tests, alpha, controls", [(1, 0.05, 50), (2, 0.05, 79), (4, 0.05, 159), (30, 0.05, 1000)])
def test_enough_controls_for_the_correction(tests, alpha, controls):
    assert controls_for_fdr(tests, alpha) == controls


# ---------------------------------------------------------------------------
# A cache with planted findings
# ---------------------------------------------------------------------------

N_SUBJECTS, SCANS = 120, 2


class TableInspector:
    """Serves precomputed layers for whichever rows were inferred."""

    extracts_hidden_layers = True

    def __init__(self, tables):
        self.tables, self.pending = tables, {}

    def attach_hooks(self, names):
        pass

    def retrieve_features(self, expected_rows=None):
        return dict(self.pending)

    def clear_buffer(self):
        self.pending = {}

    def teardown(self):
        pass


class TableConnector(ValueConnector):
    def __init__(self, inspector, outputs):
        super().__init__()
        self.inspector, self.outputs = inspector, outputs

    def load_item(self, row_id, meta):
        return int(meta["value"])

    def run_inference(self, model, batch):
        self.inspector.pending = {name: table[batch] for name, table in self.inspector.tables.items()}
        return self.outputs[batch]


def planted(seed=0, n_noise_layers=1):
    """Subjects scanned twice. The model is right 90% of the time for men, 55% for women (a fairness
    gap); `site_layer` encodes the site; the `noise*` layers encode nothing."""
    rng = np.random.default_rng(seed)
    subject = np.repeat(np.arange(N_SUBJECTS), SCANS)
    n = len(subject)
    sex = np.array(["F", "M"])[rng.integers(0, 2, N_SUBJECTS)][subject]
    site = np.array(["A", "B", "C"])[rng.integers(0, 3, N_SUBJECTS)][subject]
    label = rng.integers(0, 2, N_SUBJECTS)[subject]
    right = rng.random(N_SUBJECTS)[subject] < np.where(sex == "M", 0.9, 0.55)
    outputs = np.where(right, label, 1 - label) * 0.8 + 0.1
    tables = {"site_layer": (np.eye(3)[np.searchsorted(["A", "B", "C"], site)] * 2
                             + rng.normal(0, 0.5, (n, 3))).astype(np.float32)}
    for i in range(n_noise_layers):
        tables[f"noise{i}"] = rng.normal(size=(n, 4)).astype(np.float32)
    metadata = pd.DataFrame({"scan": [f"X{i:04d}" for i in range(n)], "value": np.arange(n),
                             "patient": [f"P{s:03d}" for s in subject], "sex": sex, "site": site, "label": label})
    return metadata, tables, outputs


def planted_cache(tmp_path, seed=0, n_noise_layers=1):
    metadata, tables, outputs = planted(seed, n_noise_layers)
    inspector = TableInspector(tables)
    run = run_engine(tmp_path / "out", metadata, TableConnector(inspector, outputs), inspector,
                     layers=list(tables), batch_size=60, id_col="scan", group_col="patient")
    assert run.ok, run.logs[-3:]
    return run.out_dir


FAST = dict(bootstrap=200, slices=["sex", "site"], concepts=["sex", "site"])


def audit(cache, **options):
    return run_audit(str(cache), AuditOptions(**{**FAST, **options}), log=lambda message: None)


# ---------------------------------------------------------------------------
# The audit finds what was planted
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def planted_audit(tmp_path_factory):
    cache = planted_cache(tmp_path_factory.mktemp("planted"))
    return cache, audit(cache)


def test_the_recorded_columns_are_used(planted_audit):
    _, result = planted_audit

    assert result["columns"] == {"id": "scan", "target": "label", "group": "patient"}
    assert result["rows"] == 240 and result["subjects"] == 120
    assert result["layers"] == ["noise0", "site_layer"]  # in the cache's (alphabetical) order


def test_the_planted_fairness_gap_is_flagged(planted_audit):
    _, result = planted_audit

    sex = result["performance"]["slices"]["sex"]
    assert sex["bootstrap"]["unit"] == "group"  # whole subjects resampled
    assert sex["gaps"]["accuracy"]["flagged"] and sex["gaps"]["accuracy"]["high"] == "M"
    assert not any(g["flagged"] for g in result["performance"]["slices"]["site"]["gaps"].values())


def test_only_the_planted_encoding_survives_the_correction(planted_audit):
    _, result = planted_audit

    probes = result["probes"]
    assert probes["n_controls"] == controls_for_fdr(4, 0.05) and probes["tested"] == 4
    found = [(r["layer"], r["concept"]) for r in probes["rows"] if r["significant"]]
    assert found == [("site_layer", "site")]
    assert all(r["q_value"] >= r["p_value"] for r in probes["rows"])


def test_the_report_leads_with_the_findings(planted_audit):
    _, result = planted_audit

    report = render_report(result)

    summary = report.split("## 1.")[0]
    assert "`site_layer` encodes `site`" in summary
    assert "`sex`: " in summary and "accuracy gap" in summary and "(M vs F)" in summary
    assert "predictive equality (FPR) gap" in summary  # acronyms keep their case
    for heading in ("## 1. Confound and leakage screen", "## 2. Performance and fairness", "### By `sex`",
                    "## 3. Linear probes", "## Settings"):
        assert heading in report
    assert "| `site_layer` | `site` |" in report and "**encoded**" in report
    assert "* Layers: `noise0` (4 values), `site_layer` (3 values)" in report
    assert "120 subjects (Group By `patient`)" in report


def test_the_json_has_no_nan(planted_audit):
    _, result = planted_audit

    text = to_json(result)

    assert "NaN" not in text  # json.loads would accept NaN, other readers would not
    assert json.loads(text)["probes"]["rows"][0]["layer"] == "noise0"


@pytest.mark.slow
def test_chance_findings_are_not_reported(tmp_path):
    """Ten layers of noise probed for sex and site: some reach p < 0.05 by chance, and the correction drops them."""
    cache = planted_cache(tmp_path, seed=0, n_noise_layers=10)

    result = audit(cache)

    rows = result["probes"]["rows"]
    lucky = [r for r in rows if r["layer"].startswith("noise") and r["p_value"] < 0.05]
    assert len(lucky) >= 2  # without a correction, these would be "findings"
    assert not any(r["significant"] for r in lucky)
    assert [(r["layer"], r["concept"]) for r in rows if r["significant"]] == [("site_layer", "site")]


# ---------------------------------------------------------------------------
# Options and problems
# ---------------------------------------------------------------------------

def test_too_few_shuffles_for_the_correction_are_explained(planted_audit):
    cache, _ = planted_audit

    result = audit(cache, controls=10)

    assert not any(r["significant"] for r in result["probes"]["rows"])
    assert "can only show p >= 0.0909" in result["probes"]["notes"][0]


def test_a_probe_that_cannot_run_is_listed_but_not_counted(planted_audit):
    cache, _ = planted_audit

    result = audit(cache, concepts=["site", "scan"], layers=["site_layer"])  # every scan is its own class

    errors = [r for r in result["probes"]["rows"] if r["result"]["error"]]
    assert [r["concept"] for r in errors] == ["scan"] and result["probes"]["tested"] == 1
    assert "not run:" in render_report(result)


def test_without_group_by_rows_are_resampled(planted_audit):
    cache, _ = planted_audit

    result = audit(cache, no_group=True, concepts=[])

    assert result["columns"]["group"] is None and result["subjects"] is None
    assert result["performance"]["global"]["bootstrap"]["unit"] == "row"


def test_default_slices_skip_ids_groups_and_the_target(planted_audit):
    cache, _ = planted_audit

    result = run_audit(str(cache), AuditOptions(bootstrap=50, concepts=[]), log=lambda m: None)

    assert result["slices"] == ["sex", "site"]


@pytest.mark.parametrize("options, message", [
    (dict(slices=["age"]), "Unknown slice column"),
    (dict(concepts=["age"]), "Unknown concept column"),
    (dict(layers=["fc"]), "Unknown layer"),
    (dict(positive="2"), "positive class '2'"),
    (dict(model="svm"), "Unknown probe model"),
    (dict(fdr=1.5), "false discovery rate"),
    (dict(group_col="family"), "Group By column 'family'"),
])
def test_unusable_options_are_explained(planted_audit, options, message):
    cache, _ = planted_audit

    with pytest.raises(AuditError, match=message):
        audit(cache, **options)


def test_rows_the_extraction_lost_lead_the_summary(tmp_path):
    metadata, tables, outputs = planted()
    inspector = TableInspector(tables)

    class LosesSiteC(TableConnector):
        def load_item(self, row_id, meta):
            if meta["site"] == "C":
                raise OSError(f"cannot read {row_id}")
            return super().load_item(row_id, meta)

    run = run_engine(tmp_path / "out", metadata, LosesSiteC(inspector, outputs), inspector, layers=list(tables),
                     batch_size=60, id_col="scan", group_col="patient", max_failed_percent=100)
    result = audit(run.out_dir, slices=["sex"], concepts=["sex"])

    report = render_report(result)
    summary = report.split("## 1.")[0]
    lost = int((metadata["site"] == "C").sum())
    assert (result["rows"], result["rows_attempted"]) == (len(metadata) - lost, len(metadata))
    assert f"**Rows left out.** {lost} of {len(metadata)} rows" in summary
    assert "Most affected: site = C (all" in summary
    assert report.count("failed during the extraction") == 1  # not repeated under Notes


def test_the_report_ends_with_what_produced_the_results(planted_audit):
    import prism
    _, result = planted_audit

    provenance = render_report(result).split("## Provenance")[1]

    assert f"* Extracted with PRISM {prism.__version__} on Python " in provenance
    assert "* Packages: numpy " in provenance and "scikit-learn " in provenance
    assert "* Metadata CSV: not recorded" in provenance  # the planted run was in-process, with no files
    assert "* Model weights: no fingerprint" in provenance  # nor a framework that can fingerprint its model
    assert f"* Report written by PRISM {prism.__version__}." in provenance


def test_a_legacy_cache_needs_its_target_named(tmp_path):
    metadata = pd.DataFrame({"subject_id": [f"S{i}" for i in range(40)], "label": [0, 1] * 20,
                             "sex": ["F"] * 20 + ["M"] * 20})
    cache = write_legacy_cache(tmp_path / "cache", metadata, np.tile([0.2, 0.8], 20))

    with pytest.raises(AuditError, match="--target"):
        run_audit(str(cache), AuditOptions(), log=lambda m: None)
    result = run_audit(str(cache), AuditOptions(target="label", bootstrap=50), log=lambda m: None)

    assert result["performance"]["global"]["global"]["metrics"]["accuracy"]["value"] == 1.0
    assert "predates row IDs" in render_report(result)
    assert "## Provenance\n\nNot recorded" in render_report(result)


def test_regression_and_multiclass_targets_are_reported(tmp_path):
    rng = np.random.default_rng(0)
    age = rng.uniform(55, 90, 60)
    metadata = pd.DataFrame({"subject_id": [f"S{i}" for i in range(60)], "age": age, "dx": list("abc") * 20,
                             "sex": ["F", "M"] * 30})
    regression = write_legacy_cache(tmp_path / "r", metadata, age + rng.normal(0, 2, 60))
    multiclass = write_legacy_cache(tmp_path / "m", metadata, np.eye(3)[[0, 1, 2] * 20])

    by_age = render_report(run_audit(str(regression), AuditOptions(target="age", bootstrap=50), log=lambda m: None))
    by_dx = render_report(run_audit(str(multiclass), AuditOptions(target="dx", bootstrap=50), log=lambda m: None))

    assert "| Group | n | Subjects | RMSE | MAE | Mean error | R² |" in by_age
    assert "Mean error (bias)" in by_age
    assert "Recall c" in by_dx and "Balanced accuracy" in by_dx


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def test_the_command_writes_the_report_next_to_the_results(planted_audit, capsys):
    cache, _ = planted_audit

    code = main([str(cache), "--bootstrap", "50", "--concept", "site", "--layer", "site_layer",
                 "--json", str(cache / "audit.json")])

    assert code == 0
    report = (cache / REPORT_NAME).read_text(encoding="utf-8")
    assert report.startswith("# PRISM audit of out")
    assert json.loads((cache / "audit.json").read_text())["concepts"] == ["site"]
    out = capsys.readouterr().out
    assert "[1/1] Probing 'site_layer' for 'site'" in out and "[SUCCESS] Report written to" in out


def test_the_command_reports_unusable_options_and_exits_2(planted_audit, capsys):
    cache, _ = planted_audit

    code = main([str(cache), "--slice", "age"])

    assert code == 2
    assert "[ERROR] Unknown slice column(s): age" in capsys.readouterr().err


def test_the_command_reports_a_missing_cache(tmp_path, capsys):
    assert main([str(tmp_path / "nowhere")]) == 2
    assert "Cache files not found" in capsys.readouterr().err
