"""Evidence tab: metadata loading, column mapping and the leakage profiler."""
import numpy as np
import pandas as pd
import pytest


pytestmark = pytest.mark.gui

RISK = 4  # column index of "Leakage Risk" in the profiler tree


def load(app, tmp_path, frame):
    path = tmp_path / "meta.csv"
    frame.to_csv(path, index=False)
    evidence = app.frame_evidence
    evidence.load_csv(str(path))
    return evidence


def profile(evidence, target):
    evidence.cb_target.set(target)
    evidence.run_leakage_check()
    rows = (evidence.tree_prof.item(i, "values") for i in evidence.tree_prof.get_children())
    return {str(values[0]): values for values in rows}


@pytest.fixture
def cohort():
    rng = np.random.default_rng(0)
    diagnosis = rng.integers(0, 2, 200)
    return pd.DataFrame({
        "subject_id": [f"S{i:03d}" for i in range(200)],
        "diagnosis": diagnosis,
        "site": np.where(diagnosis == 1, "A", "B"),
        "sex": rng.choice(["F", "M"], 200),
    })


def test_loading_a_csv_fills_the_column_pickers(app, tmp_path, cohort):
    evidence = load(app, tmp_path, cohort)

    assert list(evidence.cb_id["values"]) == list(cohort.columns)
    assert list(evidence.cb_group["values"]) == ["(None)"] + list(cohort.columns)
    assert evidence.cb_id.get() == "subject_id"  # guessed from the first column's name
    assert evidence.cb_group.get() == "(None)"
    assert len(evidence.tree_prev.get_children()) == 50
    assert app.lbl_evidence.cget("text") == "Evidence: Loaded (Unmapped)"


def test_an_unreadable_csv_is_reported(app, tmp_path, dialogs):
    app.frame_evidence.load_csv(str(tmp_path / "missing.csv"))

    assert dialogs.kinds == ["showerror"]


def test_leakage_profiler_flags_a_perfect_categorical_proxy(app, tmp_path, cohort):
    rows = profile(load(app, tmp_path, cohort), "diagnosis")

    assert rows["site"][RISK] == "EXTREME (AUC 1.00)"
    assert rows["sex"][RISK].split()[0] in ("NONE", "LOW")
    assert rows["diagnosis"][RISK] == "N/A"
    assert rows["subject_id"][RISK] == "N/A"  # the ID column is never screened


def test_risky_columns_are_coloured(app, tmp_path, cohort):
    evidence = load(app, tmp_path, cohort)
    profile(evidence, "diagnosis")

    tags = {str(evidence.tree_prof.item(i, "values")[0]): evidence.tree_prof.item(i, "tags")
            for i in evidence.tree_prof.get_children()}
    assert tags["site"] == ("danger",)
    assert tags["sex"] == ("normal",)


def test_leakage_profiler_reports_missingness(app, tmp_path, cohort):
    cohort.loc[:49, "sex"] = np.nan

    rows = profile(load(app, tmp_path, cohort), "diagnosis")

    assert rows["sex"][3] == "25.0%"


def test_confirm_mapping_requires_id_and_target(app, tmp_path, cohort, dialogs):
    evidence = load(app, tmp_path, cohort)
    evidence.cb_id.set("")

    evidence.confirm_mapping()

    assert dialogs.kinds == ["showwarning"]
    assert getattr(app, "metadata_df", None) is None


@pytest.mark.parametrize("group_choice, expected", [("(None)", None), ("site", "site")])
def test_confirm_mapping_publishes_the_mapping(app, tmp_path, cohort, dialogs, group_choice, expected):
    evidence = load(app, tmp_path, cohort)
    evidence.cb_id.set("subject_id")
    evidence.cb_target.set("diagnosis")
    evidence.cb_group.set(group_choice)

    evidence.confirm_mapping()

    assert app.metadata_df is evidence.df
    assert app.csv_path == str(tmp_path / "meta.csv")
    assert (app.config_id_col, app.config_target_col) == ("subject_id", "diagnosis")
    assert app.config_group_col == expected
    assert app.lbl_evidence.cget("text") == "Evidence: Mapped"
    assert dialogs.kinds == ["showinfo"]


def test_continuous_proxies_are_screened(app, tmp_path, cohort):
    """e.g. MMSE for an Alzheimer's label: many unique values, so the profiler never looks."""
    rng = np.random.default_rng(1)
    cohort["mmse"] = np.where(cohort["diagnosis"] == 1, rng.uniform(10, 20, 200), rng.uniform(25, 30, 200))

    rows = profile(load(app, tmp_path, cohort), "diagnosis")

    assert rows["mmse"][RISK] != "N/A"


def test_a_singleton_category_is_not_reported_as_a_proxy(app, tmp_path, cohort):
    rng = np.random.default_rng(2)
    cohort["scanner"] = rng.choice(["Siemens", "GE"], 200)
    cohort.loc[0, "scanner"] = "Philips"  # one scan from a third vendor

    rows = profile(load(app, tmp_path, cohort), "diagnosis")

    assert not rows["scanner"][RISK].startswith("EXTREME")


def test_group_by_is_respected_and_changing_it_re_screens(app, tmp_path):
    rng = np.random.default_rng(3)
    per_subject = rng.integers(0, 2, 40)
    cohort = pd.DataFrame({
        "scan_id": [f"S{i}" for i in range(480)],
        "subject": np.repeat([f"P{i:03d}" for i in range(40)], 12),
        "diagnosis": np.repeat(per_subject, 12),
    })
    evidence = load(app, tmp_path, cohort)
    evidence.cb_id.set("scan_id")

    assert profile(evidence, "diagnosis")["subject"][RISK].startswith("EXTREME")  # no Group By yet
    evidence.cb_group.set("subject")
    assert profile(evidence, "diagnosis")["subject"][RISK] == "N/A"
    assert evidence.cb_group.bind("<<ComboboxSelected>>") and evidence.cb_id.bind("<<ComboboxSelected>>")


@pytest.fixture
def zero_padded():
    return pd.DataFrame({"subject_id": ["0042", "0043", "0100", "1e3"], "patient": ["007", "007", "012", "012"],
                         "label": [0, 1, 0, 1]})


def test_ids_with_leading_zeros_are_kept_in_the_preview_and_the_mapping(app, tmp_path, dialogs, zero_padded):
    # debt P1 #25
    evidence = load(app, tmp_path, zero_padded)  # subject_id is guessed as the ID column
    preview = [evidence.tree_prev.item(i, "values")[0] for i in evidence.tree_prev.get_children()]
    assert preview == ["0042", "0043", "0100", "1e3"]

    evidence.cb_target.set("label")
    evidence.confirm_mapping()

    assert list(app.metadata_df["subject_id"]) == ["0042", "0043", "0100", "1e3"]
    assert app.metadata_df["label"].dtype.kind == "i"


def test_the_id_column_chosen_by_hand_is_read_as_text(app, tmp_path, dialogs, zero_padded):
    evidence = load(app, tmp_path, zero_padded[["label", "patient", "subject_id"]])  # no guess: "label" first
    assert evidence.cb_id.get() == ""

    evidence.cb_id.set("subject_id")
    evidence.on_id_selected()
    preview = [evidence.tree_prev.item(i, "values")[2] for i in evidence.tree_prev.get_children()]
    evidence.cb_id.set("patient")  # set without the event, as a test or a script might
    evidence.cb_target.set("label")
    evidence.confirm_mapping()

    assert preview == ["0042", "0043", "0100", "1e3"]
    assert list(app.metadata_df["patient"]) == ["007", "007", "012", "012"]
    assert app.metadata_df["subject_id"].tolist() == [42.0, 43.0, 100.0, 1000.0]  # no longer the ID column
