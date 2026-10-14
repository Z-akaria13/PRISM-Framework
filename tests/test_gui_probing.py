"""Probing tab: drives analysis.linear_probe on cached layers and reports the result."""
import re

import numpy as np
import pandas as pd
import pytest

from _support import TensorConnector, known_bug, run_engine, wait_for, write_legacy_cache

pytestmark = pytest.mark.gui


def probing_cache(tmp_path, n=60, seed=0):
    rng = np.random.default_rng(seed)
    sex = rng.choice(["F", "M"], n)
    age = rng.uniform(55, 90, n)
    metadata = pd.DataFrame({
        "subject_id": [f"S{i:03d}" for i in range(n)],
        "patient": np.repeat(np.arange(n // 3), 3),
        "sex": sex,
        "age": age,
    })
    layers = {
        "encodes_sex": (sex == "F")[:, None] * 2.0 + rng.normal(0, 0.1, (n, 4)),
        "encodes_age": ((age - 70) / 10)[:, None] + rng.normal(0, 0.05, (n, 4)),
        "noise": rng.normal(size=(n, 4)),
    }
    return metadata, write_legacy_cache(tmp_path / "cache", metadata, np.zeros(n), layers)


def open_tab(app, cache_dir):
    app.active_cache_dir = str(cache_dir)
    tab = app.frame_probing
    tab.load_options()
    return tab


def probe(tab, layer, concept, control=True):
    tab.cb_layer.set(layer)
    tab.cb_concept.set(concept)
    tab.var_control.set(control)
    task = tab.run_probe()
    if task is not None:
        wait_for(tab.app, task)
    return tab.txt_res.get("1.0", "end")


def score(output, label):
    return float(re.search(re.escape(label) + r": (-?[0-9.]+)", output).group(1))


def test_run_probe_says_what_is_missing(app, tmp_path, dialogs):
    tab = app.frame_probing
    tab.run_probe()  # nothing loaded
    metadata, cache = probing_cache(tmp_path)
    open_tab(app, cache)
    tab.cb_concept.set("")
    tab.run_probe()  # no concept chosen
    black_box = write_legacy_cache(tmp_path / "black_box", metadata, np.zeros(len(metadata)))
    open_tab(app, black_box)
    tab.run_probe()  # results without layers, e.g. from a scikit-learn model

    assert [(kind, title) for kind, title, _ in dialogs.calls] == [
        ("showinfo", "No Results"), ("showinfo", "Nothing Chosen"), ("showinfo", "No Layers")]
    assert tab.task is None


def test_loading_the_cache_lists_layers_and_concepts(app, tmp_path):
    metadata, cache = probing_cache(tmp_path)

    tab = open_tab(app, cache)

    assert set(tab.cb_layer["values"]) == {"encodes_sex", "encodes_age", "noise"}
    assert list(tab.cb_concept["values"]) == list(metadata.columns)


def test_probe_detects_an_encoded_attribute_and_runs_the_control(app, tmp_path):
    _, cache = probing_cache(tmp_path)

    out = probe(open_tab(app, cache), "encodes_sex", "sex")

    assert "Detected Task: Classification (Shrinkage LDA, streamed)" in out
    assert score(out, "RESULT (AUC)") >= 0.95
    assert "chance: 0.500" in out
    control_auc = float(re.search(r"Control \(50 label shuffles, per row\): AUC ([0-9.]+)", out).group(1))
    assert control_auc < 0.75
    assert "0 of 50 shuffles matched or beat the real probe (p = 0.020)" in out
    assert "WARNING" not in out


def test_probe_on_noise_is_near_chance(app, tmp_path):
    _, cache = probing_cache(tmp_path)

    out = probe(open_tab(app, cache), "noise", "sex", control=False)

    assert score(out, "RESULT (AUC)") < 0.75
    assert "Control" not in out


def test_numeric_concept_uses_a_ridge_regression_probe(app, tmp_path):
    _, cache = probing_cache(tmp_path)

    out = probe(open_tab(app, cache), "encodes_age", "age", control=False)

    assert "Detected Task: Regression (Ridge)" in out
    assert score(out, "RESULT (R² Score)") > 0.9


def spy_on_probe(monkeypatch):
    """Record the groups the Probing tab hands to analysis.linear_probe."""
    import prism.gui.frames.probing as probing_module
    seen = []
    real_probe = probing_module.linear_probe

    def spy(*args, **kwargs):
        seen.append(kwargs.get("groups"))
        return real_probe(*args, **kwargs)

    monkeypatch.setattr(probing_module, "linear_probe", spy)
    return seen


def test_group_by_keeps_each_group_in_one_fold(app, tmp_path, monkeypatch):
    _, cache = probing_cache(tmp_path)
    app.config_group_col = "patient"
    seen = spy_on_probe(monkeypatch)

    out = probe(open_tab(app, cache), "encodes_sex", "sex", control=False)

    assert seen[0].nunique() == 20
    assert "Folds keep each 'patient' group together (20 groups)." in out
    assert "No Group By column is set" not in out


def test_missing_group_by_is_flagged(app, tmp_path):
    _, cache = probing_cache(tmp_path)

    out = probe(open_tab(app, cache), "encodes_sex", "sex", control=False)

    assert "[NOTE] No Group By column is set" in out


def test_a_probe_that_does_not_beat_chance_is_flagged(app, tmp_path):
    _, cache = probing_cache(tmp_path)

    out = probe(open_tab(app, cache), "noise", "sex")

    assert "WARNING: the real probe does not clearly beat shuffled labels" in out


def test_too_few_samples_in_a_class_is_refused(app, tmp_path, dialogs):
    metadata, _ = probing_cache(tmp_path)
    metadata["rare"] = ["x"] + ["y"] * (len(metadata) - 1)
    cache = write_legacy_cache(tmp_path / "rare_cache", metadata, np.zeros(len(metadata)),
                        {"noise": np.zeros((len(metadata), 2))})

    probe(open_tab(app, cache), "noise", "rare")

    assert "too few samples" in dialogs.messages("showwarning")[0]


def test_a_missing_label_does_not_block_the_probe(app, tmp_path, dialogs):
    metadata, _ = probing_cache(tmp_path)
    metadata.loc[0, "sex"] = np.nan
    layers = {"encodes_sex": (metadata["sex"] == "F").to_numpy()[:, None] * 2.0}
    cache = write_legacy_cache(tmp_path / "nan_cache", metadata, np.zeros(len(metadata)), layers)

    out = probe(open_tab(app, cache), "encodes_sex", "sex", control=False)

    assert "RESULT (AUC)" in out
    assert "[NOTE] 1 rows with a missing target were left out." in out
    assert dialogs.kinds == []


def engine_cache(tmp_path, group_col):
    """A cache in the current format, extracted with Group By = `group_col`."""
    torch = pytest.importorskip("torch")
    from prism.inspectors.pytorch import PyTorchInspector
    metadata, _ = probing_cache(tmp_path)
    metadata["value"] = range(len(metadata))
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(3, 4)).eval()
    run = run_engine(tmp_path / "extracted", metadata, TensorConnector((3,)), PyTorchInspector(model),
                     model=model, layers=["0"], target_col="sex", group_col=group_col)
    assert run.ok
    return run.out_dir


def test_group_by_recorded_with_the_cache_is_used_in_a_new_session(app, tmp_path, monkeypatch):
    seen = spy_on_probe(monkeypatch)
    tab = open_tab(app, engine_cache(tmp_path, group_col="patient"))  # nothing confirmed in Evidence

    out = probe(tab, "0", "sex", control=False)

    assert "folds keep each 'patient' group together" in out
    assert seen[0] is not None and seen[0].name == "patient"


def test_this_sessions_evidence_mapping_takes_precedence(app, tmp_path, monkeypatch):
    seen = spy_on_probe(monkeypatch)
    app.metadata_df, app.config_group_col = pd.DataFrame(), None  # Evidence confirmed with Group By "(None)"
    tab = open_tab(app, engine_cache(tmp_path, group_col="patient"))

    probe(tab, "0", "sex", control=False)

    assert seen[0] is None


def test_misaligned_layers_from_old_caches_are_left_out(app, tmp_path):
    metadata, _ = probing_cache(tmp_path)
    layers = {"aligned": np.zeros((60, 2)), "short": np.zeros((57, 2))}
    cache = write_legacy_cache(tmp_path / "old", metadata, np.zeros(60), layers)

    tab = open_tab(app, cache)

    assert list(tab.cb_layer["values"]) == ["aligned"]
    assert "Layer 'short' has 57 rows for 60 metadata rows" in tab.txt_res.get("1.0", "end")


def test_shrinkage_lda_is_the_default_probe(app, tmp_path):
    _, cache = probing_cache(tmp_path)

    assert open_tab(app, cache).var_model.get() == "lda"


@pytest.mark.parametrize("model, label", [
    ("lda", "Classification (Shrinkage LDA, streamed)"),
    ("logistic", "Classification (LogReg)"),
    ("ridge", "Classification (Ridge classifier, streamed)"),
])
def test_every_probe_model_can_be_chosen(app, tmp_path, model, label):
    _, cache = probing_cache(tmp_path)
    tab = open_tab(app, cache)
    tab.var_model.set(model)

    out = probe(tab, "encodes_sex", "sex", control=False)

    assert f"Detected Task: {label}" in out
    assert score(out, "RESULT (AUC)") >= 0.95


def test_the_ridge_probe_streams_the_layer_from_the_cache(app, tmp_path, monkeypatch):
    import h5py
    import prism.gui.frames.probing as probing_module
    _, cache = probing_cache(tmp_path)
    calls = []
    real_probe = probing_module.linear_probe

    def spy(features, *args, **kwargs):
        calls.append((type(features), kwargs.get("model")))
        return real_probe(features, *args, **kwargs)

    monkeypatch.setattr(probing_module, "linear_probe", spy)
    tab = open_tab(app, cache)
    tab.var_model.set("ridge")

    out = probe(tab, "encodes_sex", "sex")

    assert calls == [(h5py.Dataset, "ridge")]  # handed the on-disk dataset, not a loaded array
    assert "Detected Task: Classification (Ridge classifier, streamed)" in out
    assert score(out, "RESULT (AUC)") >= 0.95
    assert "0 of 50 shuffles matched or beat the real probe" in out


@pytest.mark.parametrize("model, warned", [("logistic", True), ("lda", False), ("ridge", False)])
def test_only_the_in_memory_probe_warns_about_large_layers(app, tmp_path, dialogs, monkeypatch, model, warned):
    _, cache = probing_cache(tmp_path)
    tab = open_tab(app, cache)
    monkeypatch.setattr(tab.cache, "layer_nbytes", lambda name: 5 * 2**30)
    tab.var_model.set(model)

    probe(tab, "encodes_sex", "sex", control=False)

    assert ("Memory Warning" in [title for _, title, _ in dialogs.calls]) is warned


def test_shuffles_default_to_fifty_and_show_the_lowest_reachable_p(app, tmp_path):
    _, cache = probing_cache(tmp_path)
    tab = open_tab(app, cache)

    assert tab.var_shuffles.get() == "50"
    assert tab.lbl_min_p.cget("text") == "lowest possible p: 0.020"
    tab.var_shuffles.set("999")
    assert tab.lbl_min_p.cget("text") == "lowest possible p: 0.001"


def test_the_chosen_number_of_shuffles_is_used(app, tmp_path):
    _, cache = probing_cache(tmp_path)
    tab = open_tab(app, cache)
    tab.var_shuffles.set("199")

    out = probe(tab, "encodes_sex", "sex")

    assert "Control (199 label shuffles, per row)" in out
    assert "0 of 199 shuffles matched or beat the real probe (p = 0.005)" in out


@pytest.mark.parametrize("entry", ["0", "1001", "lots", ""])
def test_an_invalid_number_of_shuffles_is_refused(app, tmp_path, dialogs, entry):
    _, cache = probing_cache(tmp_path)
    tab = open_tab(app, cache)
    tab.var_shuffles.set(entry)

    out = probe(tab, "encodes_sex", "sex")

    assert "Shuffles must be a whole number from 1 to 1000" in dialogs.messages("showerror")[0]
    assert "RESULT" not in out
    assert tab.lbl_min_p.cget("text") == "enter 1 to 1000"


def test_shuffles_are_ignored_when_the_control_is_off(app, tmp_path, dialogs):
    _, cache = probing_cache(tmp_path)
    tab = open_tab(app, cache)
    tab.var_shuffles.set("lots")

    out = probe(tab, "encodes_sex", "sex", control=False)

    assert "RESULT (AUC)" in out and dialogs.kinds == []


def test_each_call_of_a_reused_module_and_all_of_them_can_be_probed(app, tmp_path):
    torch = pytest.importorskip("torch")
    from prism.inspectors.pytorch import PyTorchInspector

    class Reused(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.lin = torch.nn.Linear(3, 3)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            return self.relu(self.lin(self.relu(x)))

    metadata, _ = probing_cache(tmp_path)
    metadata["value"] = range(len(metadata))
    model = Reused().eval()
    run = run_engine(tmp_path / "extracted", metadata, TensorConnector((3,)), PyTorchInspector(model),
                     model=model, layers=["relu"], target_col="sex")
    tab = open_tab(app, run.out_dir)

    assert list(tab.cb_layer["values"]) == ["relu@0", "relu@1", "relu@all"]
    out = probe(tab, "relu@all", "sex", control=False)
    assert "Data: (60, 6)" in out and "RESULT (AUC)" in out



def test_a_running_probe_can_be_cancelled(app, tmp_path, monkeypatch):
    import threading
    import prism.gui.frames.probing as probing_module
    from prism.analysis import linear_probe as real_probe
    _, cache = probing_cache(tmp_path)
    tab = open_tab(app, cache)
    started = threading.Event()

    def slow_probe(*args, progress=None, **kwargs):
        started.set()
        for _ in range(500):
            threading.Event().wait(0.01)
            progress(0.1)  # raises once cancelled
        return real_probe(*args, **kwargs)

    monkeypatch.setattr(probing_module, "linear_probe", slow_probe)
    tab.cb_layer.set("encodes_sex")
    tab.cb_concept.set("sex")
    task = tab.run_probe()
    assert started.wait(5)
    assert str(tab.btn_run.cget("state")) == "disabled"

    tab.cancel_probe()
    wait_for(app, task)

    assert "[CANCELLED] Probe stopped." in tab.txt_res.get("1.0", "end")
    assert "RESULT" not in tab.txt_res.get("1.0", "end")
    assert str(tab.btn_run.cget("state")) == "normal"


def test_the_probe_progress_bar_reaches_the_end(app, tmp_path):
    _, cache = probing_cache(tmp_path)

    probe(open_tab(app, cache), "encodes_sex", "sex")

    assert app.frame_probing.progress_var.get() == pytest.approx(1.0)
