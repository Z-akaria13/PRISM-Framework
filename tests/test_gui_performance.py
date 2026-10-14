"""Performance tab: task detection, global metrics, fairness slicing, and parity with analysis.py."""
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from _support import ValueConnector, run_engine, value_frame, wait_idle, write_legacy_cache
from prism.analysis import subgroup_performance
from prism.inspectors.sklearn import SklearnInspector

pytestmark = pytest.mark.gui

# Same hand-checked example as test_analysis.py.
PROBS = [0.9, 0.2, 0.8, 0.4, 0.7, 0.1]
LABELS = [1, 0, 1, 1, 0, 0]
GROUPS = list("AAABBB")


def binary_cache(tmp_path, probs=PROBS, labels=LABELS, groups=GROUPS):
    metadata = pd.DataFrame({"subject_id": [f"S{i:03d}" for i in range(len(labels))],
                             "label": labels, "group": groups})
    return write_legacy_cache(tmp_path / "cache", metadata, probs)


def open_tab(app, cache_dir, target="label", min_group="1"):
    """Opens the tab with Min Group Size 1 unless told otherwise, so the six-row examples are compared."""
    app.active_cache_dir = str(cache_dir)
    app.config_target_col = target
    tab = app.frame_performance
    tab.var_min_group.set(min_group)
    tab.load_results()
    wait_idle(app, tab.runner)
    return tab


def analyse(tab):
    """Run the analysis for the current controls and wait for the background bootstrap."""
    tab.run_analysis()
    wait_idle(tab.app, tab.runner)


def sliced(tab, column):
    tab.cb_slice.set(column)
    analyse(tab)
    return report(tab)


def report(tab):
    return tab.txt_report.get("1.0", "end")


def reported(tab, label):
    return float(re.search(re.escape(label) + r"\s*:\s*([0-9.]+)", report(tab)).group(1))


def test_binary_target_global_report(app, tmp_path):
    tab = open_tab(app, binary_cache(tmp_path))

    assert tab.task_type == "classification"
    assert tab.cb_pos_class.get() == "1"
    assert list(tab.cb_slice["values"]) == ["(Global)", "subject_id", "group"]
    assert tab.lbl_stats.cget("text").startswith("Global Acc: 66.7%")
    assert reported(tab, "Sensitivity") == pytest.approx(0.667)
    assert reported(tab, "FPR (Alarm)") == pytest.approx(0.333)
    assert "True Positives  (Hit)   : 2" in report(tab)


def test_fairness_slice_reports_per_group_rates_and_gaps(app, tmp_path):
    tab = open_tab(app, binary_cache(tmp_path))
    tab.cb_slice.set("group")

    analyse(tab)

    text = report(tab)
    assert "--- FAIRNESS REPORT: GROUP ---" in text
    assert "[A] (n=3)" in text and "[B] (n=3)" in text
    assert reported(tab, "Equal Opportunity Diff (TPR)") == pytest.approx(1.0)
    assert reported(tab, "Predictive Equality Diff (FPR)") == pytest.approx(0.5)
    assert reported(tab, "Demographic Parity Diff (Sel)") == pytest.approx(0.333, abs=1e-3)
    assert "[WARNING] High disparity in True Positive Rates." in text


def test_fairness_gaps_match_the_headless_analysis(app, tmp_path):
    tab = open_tab(app, binary_cache(tmp_path))
    tab.cb_slice.set("group")
    analyse(tab)

    gaps = subgroup_performance(np.array(PROBS), np.array(LABELS),
                                pd.DataFrame({"group": GROUPS}), "group", min_group_size=1)["gaps"]

    assert reported(tab, "Equal Opportunity Diff (TPR)") == pytest.approx(gaps["eo_tpr"], abs=1e-3)
    assert reported(tab, "Predictive Equality Diff (FPR)") == pytest.approx(gaps["eo_fpr"], abs=1e-3)
    assert reported(tab, "Demographic Parity Diff (Sel)") == pytest.approx(gaps["demographic_parity"], abs=1e-3)


def test_group_without_positives_reports_tpr_as_not_available(app, tmp_path):
    tab = open_tab(app, binary_cache(tmp_path, labels=[1, 0, 1, 0, 0, 0]))
    tab.cb_slice.set("group")

    analyse(tab)

    assert "Equal Opportunity Diff: N/A" in report(tab)


def test_numeric_target_with_many_values_is_treated_as_regression(app, tmp_path):
    y = np.arange(20, dtype=float)
    metadata = pd.DataFrame({"age": y, "group": ["A", "B"] * 10})
    tab = open_tab(app, write_legacy_cache(tmp_path / "cache", metadata, y + 0.5), target="age")

    assert tab.task_type == "regression"
    assert str(tab.cb_pos_class.cget("state")) == "disabled"
    assert tab.lbl_stats.cget("text") == "Global R²: 0.992 | RMSE: 0.500"

    text = sliced(tab, "group")
    assert "[A] (n=10)" in text
    assert reported(tab, "RMSE") == pytest.approx(0.5)
    assert reported(tab, "RMSE Diff") == pytest.approx(0.0)
    assert "Mean Error: +0.500 [+0.500, +0.500]" in text


def test_multiclass_target_gets_a_global_confusion_report(app, tmp_path):
    labels = ["a", "b", "c", "a", "b", "c"]
    preds = np.eye(3)[[0, 1, 2, 0, 1, 0]]  # last item wrong
    metadata = pd.DataFrame({"label": labels, "group": GROUPS})

    tab = open_tab(app, write_legacy_cache(tmp_path / "cache", metadata, preds))

    assert tab.task_type == "multiclass"
    assert "--- MULTICLASS GLOBAL REPORT ---" in report(tab)
    assert reported(tab, "Accuracy") == pytest.approx(0.833)


def test_multiclass_targets_can_be_sliced(app, tmp_path, dialogs):
    # Group A: all right. Group B: its "b" is called "a".
    metadata = pd.DataFrame({"label": ["a", "b", "c"] * 2, "group": GROUPS})
    tab = open_tab(app, write_legacy_cache(tmp_path / "cache", metadata, np.eye(3)[[0, 1, 2, 0, 0, 2]]))

    text = sliced(tab, "group")

    assert not dialogs.messages("showinfo")
    assert "--- MULTICLASS FAIRNESS REPORT: GROUP ---" in text
    assert "'b': Recall=0.000" in text
    assert reported(tab, "Recall Diff ('b')") == pytest.approx(1.0)
    assert "(A vs B)" in text
    assert reported(tab, "Accuracy Diff") == pytest.approx(1 / 3, abs=1e-3)


def test_missing_cache_is_reported_and_forgotten(app, tmp_path, dialogs):
    app.active_cache_dir = str(tmp_path)

    app.frame_performance.load_results()

    assert "Cache files not found" in dialogs.messages("showerror")[0]
    assert app.active_cache_dir is None


def test_rerunning_the_analysis_does_not_leak_figures(app, tmp_path):
    plt.close("all")
    tab = open_tab(app, binary_cache(tmp_path))

    for _ in range(5):
        analyse(tab)

    assert len(plt.get_fignums()) <= 1


def test_gui_and_headless_analysis_use_the_same_decision_threshold(app, tmp_path):
    probs, labels = [0.5, 0.5, 0.2, 0.9], [1, 1, 0, 1]
    tab = open_tab(app, binary_cache(tmp_path, probs=probs, labels=labels, groups=list("AABB")))

    headless = subgroup_performance(np.array(probs), np.array(labels),
                                    pd.DataFrame({"group": list("AABB")}), "group")

    gui_accuracy = float(re.search(r"Global Acc: ([0-9.]+)%", tab.lbl_stats.cget("text")).group(1)) / 100
    assert gui_accuracy == pytest.approx(headless["global"]["accuracy"])


# ---------------------------------------------------------------------------
# Caches in the current format: recorded mapping, class order, integrity
# ---------------------------------------------------------------------------

TRAINING_ORDER = ["CN", "MCI", "AD"]  # alphabetical order would be AD, CN, MCI


class OneHotConnector(ValueConnector):
    """Always predicts the right class, as one-hot scores in training order."""

    def run_inference(self, model, batch):
        return np.eye(3)[[int(v) % 3 for v in batch]]


def three_class_cache(tmp_path, declared):
    metadata = value_frame(6).assign(label=[TRAINING_ORDER[i % 3] for i in range(6)])
    connector = OneHotConnector()
    if declared:
        connector.class_names = TRAINING_ORDER
    return run_engine(tmp_path / "out", metadata, connector, SklearnInspector(None), target_col="label").out_dir


def open_in_new_session(app, cache_dir):
    """Open a cache without confirming anything in the Evidence tab first."""
    app.active_cache_dir = str(cache_dir)
    tab = app.frame_performance
    tab.load_results()
    wait_idle(app, tab.runner)
    return tab


def test_declared_class_names_decode_multiclass_outputs(app, tmp_path):
    tab = open_in_new_session(app, three_class_cache(tmp_path, declared=True))

    assert tab.target_col == "label"  # recorded with the cache: no prompt
    assert tab.unique_classes == TRAINING_ORDER
    assert reported(tab, "Accuracy") == pytest.approx(1.0)
    assert "[NOTE]" not in report(tab)


def test_an_undeclared_class_order_is_flagged_in_the_report(app, tmp_path):
    tab = open_in_new_session(app, three_class_cache(tmp_path, declared=False))

    assert "does not declare class_names" in report(tab)


def test_legacy_caches_are_flagged_in_the_report(app, tmp_path):
    tab = open_tab(app, binary_cache(tmp_path))

    assert "[NOTE] This cache predates row IDs" in report(tab)


def test_an_edited_metadata_file_is_refused(app, tmp_path, dialogs):
    out = three_class_cache(tmp_path, declared=True)
    csv = out / "prism_metadata.csv"
    pd.read_csv(csv).iloc[::-1].to_csv(csv, index=False)

    open_in_new_session(app, out)

    assert "no longer matches the row IDs" in dialogs.messages("showerror")[0]
    assert app.active_cache_dir is None


# ---------------------------------------------------------------------------
# Intervals and minimum group size
# ---------------------------------------------------------------------------

def rate_cache(tmp_path, groups, patients=None):
    """{group: (found, missed)} positives, plus four correctly rejected negatives per group."""
    labels, probs, slices = [], [], []
    for name, (found, missed) in groups.items():
        labels += [1] * (found + missed) + [0] * 4
        probs += [0.9] * found + [0.1] * missed + [0.1] * 4
        slices += [name] * (found + missed + 4)
    metadata = pd.DataFrame({"subject_id": [f"S{i:03d}" for i in range(len(labels))], "label": labels,
                             "group": slices, **({"patient": patients} if patients is not None else {})})
    return write_legacy_cache(tmp_path / "cache", metadata, probs)


def test_every_figure_carries_an_interval(app, tmp_path):
    tab = open_tab(app, binary_cache(tmp_path))

    assert "Intervals     : 95% bootstrap, 1000 resamples of 6 scans" in report(tab)
    assert re.search(r"Sensitivity : 0\.667 \[[0-9.]+, [0-9.]+\]", report(tab))

    text = sliced(tab, "group")
    assert re.search(r"Equal Opportunity Diff \(TPR\) : 1\.000 \[1\.000, 1\.000\]  \(A vs B\)", text)


def test_small_groups_are_reported_but_not_compared_by_default(app, tmp_path):
    tab = open_tab(app, binary_cache(tmp_path), min_group="")  # the box left empty: default 20

    text = sliced(tab, "group")

    assert tab.min_group_size() == 20
    assert "[A] (n=3)  [below 20: not in the disparity scores]" in text
    assert "Equal Opportunity Diff: N/A (fewer than two groups of at least 20 scans with a defined TPR)" in text
    assert "[WARNING]" not in text


def test_a_large_gap_that_may_be_chance_is_not_a_warning(app, tmp_path):
    tab = open_tab(app, rate_cache(tmp_path, {"A": (12, 8), "B": (8, 12)}), min_group="20")

    text = sliced(tab, "group")

    assert reported(tab, "Equal Opportunity Diff (TPR)") == pytest.approx(0.2)
    assert "[NOTE] Above 0.15, but the interval includes 0: this gap may be noise." in text
    assert "[WARNING]" not in text


def test_a_large_gap_between_large_groups_is_a_warning(app, tmp_path):
    tab = open_tab(app, rate_cache(tmp_path, {"A": (180, 20), "B": (100, 100)}), min_group="20")

    text = sliced(tab, "group")

    assert "[WARNING] High disparity in True Positive Rates." in text


def test_with_group_by_the_bootstrap_resamples_subjects(app, tmp_path):
    patients = [f"P{i // 4}" for i in range(88)]  # four scans per person
    app.config_group_col = "patient"

    tab = open_tab(app, rate_cache(tmp_path, {"A": (30, 10), "B": (20, 20)}, patients), min_group="10")
    text = sliced(tab, "group")

    assert "resamples of 22 subjects (Group By: patient)" in text
    assert "[A] (n=44) from 11 subjects" in text
    assert "Min group size: 10 subjects" in text
    assert "patient" not in tab.cb_slice["values"]  # people are not a demographic slice


def test_the_id_column_is_not_offered_as_a_slice(app, tmp_path):
    app.config_id_col = "subject_id"

    tab = open_tab(app, binary_cache(tmp_path))

    assert list(tab.cb_slice["values"]) == ["(Global)", "group"]


def test_changing_the_minimum_group_size_reruns_the_analysis(app, tmp_path):
    tab = open_tab(app, binary_cache(tmp_path))
    sliced(tab, "group")
    tab.var_min_group.set("4")

    analyse(tab)

    assert "Equal Opportunity Diff: N/A" in report(tab)


def test_the_minimum_group_size_box_is_clamped(app):
    tab = app.frame_performance

    for typed, used in [("0", 1), ("5000", 1000), ("abc", 20), (" 7 ", 7)]:
        tab.var_min_group.set(typed)
        assert tab.min_group_size() == used


def test_a_group_by_column_missing_from_the_results_is_noted(app, tmp_path):
    app.config_group_col = "patient"

    tab = open_tab(app, binary_cache(tmp_path))

    assert "[NOTE] The Group By column 'patient' is not in these results" in report(tab)
    assert "resamples of 6 scans" in report(tab)



def test_a_regression_model_with_several_outputs_per_row_is_refused(app, tmp_path, dialogs):
    y = np.arange(20, dtype=float)
    metadata = pd.DataFrame({"age": y, "group": ["A", "B"] * 10})

    open_tab(app, write_legacy_cache(tmp_path / "cache", metadata, np.column_stack([y, y])), target="age")

    assert "one prediction per row" in dialogs.messages("showerror")[0]


def test_the_bootstrap_runs_off_the_tk_thread(app, tmp_path, monkeypatch):
    import threading
    import prism.gui.frames.performance as performance_module
    real_audit = performance_module.fairness_audit
    threads = []

    def recording_audit(*args, **kwargs):
        threads.append(threading.current_thread() is threading.main_thread())
        return real_audit(*args, **kwargs)

    monkeypatch.setattr(performance_module, "fairness_audit", recording_audit)

    tab = open_tab(app, binary_cache(tmp_path))

    assert threads == [False]
    assert tab.lbl_stats.cget("text").startswith("Global Acc:")
