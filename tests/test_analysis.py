"""analysis.py: the Tk-free confound screen, subgroup metrics, linear probe and report."""
import math

import numpy as np
import pandas as pd
import pytest


from prism.analysis import (DISPARITY_THRESHOLD, MIN_GROUP_SIZE, _band_for, confound_screen, fairness_audit,
                      format_audit_report, linear_probe, subgroup_performance)

# Six predictions with hand-checked metrics (shared with the Performance tab tests):
#   group A: labels 1 0 1, preds 1 0 1  -> acc 1,   TPR 1, FPR 0,   positive rate 2/3
#   group B: labels 1 0 0, preds 0 1 0  -> acc 1/3, TPR 0, FPR 1/2, positive rate 1/3
PROBS = np.array([0.9, 0.2, 0.8, 0.4, 0.7, 0.1])
LABELS = np.array([1, 0, 1, 1, 0, 0])
GROUPS = pd.DataFrame({"group": list("AAABBB")})


def by_field(rows):
    return {row["field"]: row for row in rows}


# ---------------------------------------------------------------------------
# Confound screen
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("auc, band", [
    (1.0, "EXTREME"), (0.95, "EXTREME"), (0.949, "HIGH"), (0.80, "HIGH"),
    (0.65, "MEDIUM"), (0.55, "LOW"), (0.549, "NONE"), (0.5, "NONE"),
])
def test_risk_bands(auc, band):
    assert _band_for(auc) == band


@pytest.fixture
def screening_frame():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    return pd.DataFrame({
        "subject_id": [f"S{i}" for i in range(200)],
        "diagnosis": y,
        "proxy": y * 1.0,
        "inverse_proxy": 1.0 - y,
        "site": np.where(y == 1, "A", "B"),
        "noise": rng.normal(size=200),
        "constant_text": "same",
        "constant_number": 7.0,
    })


def test_confound_screen_flags_direct_and_inverse_proxies(screening_frame):
    rows = by_field(confound_screen(screening_frame, "diagnosis", id_cols=["subject_id"]))

    assert rows["proxy"]["risk"] == "EXTREME" and rows["proxy"]["auc"] == 1.0
    assert rows["inverse_proxy"]["risk"] == "EXTREME"
    assert rows["site"]["risk"] == "EXTREME"


def test_confound_screen_scores_noise_near_chance(screening_frame):
    noise = by_field(confound_screen(screening_frame, "diagnosis"))["noise"]

    assert max(noise["auc"], 1 - noise["auc"]) < 0.65


def test_confound_screen_explains_skipped_fields(screening_frame):
    rows = by_field(confound_screen(screening_frame, "diagnosis", id_cols=["subject_id"]))

    assert rows["diagnosis"]["reason"] == "target column"
    assert rows["subject_id"]["reason"] == "identifier"
    assert rows["constant_text"]["reason"] == "single value"
    assert rows["constant_number"]["reason"] == "single value"
    assert all(rows[f]["risk"] == "N/A" for f in ("diagnosis", "subject_id", "constant_text"))


def test_confound_screen_rejects_unknown_target(screening_frame):
    with pytest.raises(ValueError, match="not in metadata"):
        confound_screen(screening_frame, "missing")


def test_confound_screen_handles_single_class_target():
    rows = confound_screen(pd.DataFrame({"y": [1, 1, 1], "x": [1, 2, 3]}), "y")

    assert rows == [{"field": "y", "risk": "N/A", "auc": None, "kind": None,
                     "reason": "target has fewer than 2 classes"}]


def test_confound_screen_still_scores_columns_with_missing_values(screening_frame):
    frame = screening_frame.copy()
    frame.loc[0, "proxy"] = np.nan

    assert by_field(confound_screen(frame, "diagnosis"))["proxy"]["risk"] == "EXTREME"


def test_confound_screen_supports_multiclass_targets():
    y = np.tile([0, 1, 2], 30)
    frame = pd.DataFrame({"stage": y, "proxy": y * 1.0})

    assert by_field(confound_screen(frame, "stage"))["proxy"]["risk"] != "N/A"


def screen(frame, target="diagnosis", **kwargs):
    return by_field(confound_screen(frame, target, **kwargs))


@pytest.fixture
def diagnosis():
    return np.random.default_rng(10).integers(0, 2, 240)


def test_a_continuous_proxy_is_flagged(diagnosis):
    """e.g. MMSE for an Alzheimer's label, which the old Evidence tab never scored."""
    rng = np.random.default_rng(11)
    mmse = np.where(diagnosis == 1, rng.uniform(10, 22, 240), rng.uniform(24, 30, 240))
    mmse[:12] = np.nan

    row = screen(pd.DataFrame({"diagnosis": diagnosis, "mmse": mmse}))["mmse"]

    assert (row["risk"], row["kind"]) == ("EXTREME", "continuous")
    assert "12 missing rows ignored" in row["reason"]


def test_a_low_value_predicting_the_class_counts_as_much_as_a_high_one(diagnosis):
    rng = np.random.default_rng(12)
    score = np.where(diagnosis == 1, rng.normal(0, 1, 240), rng.normal(5, 1, 240))

    assert screen(pd.DataFrame({"diagnosis": diagnosis, "score": score}))["score"]["risk"] == "EXTREME"


def test_a_single_unusual_category_is_not_a_proxy(diagnosis):
    rng = np.random.default_rng(13)
    scanner = rng.choice(["Siemens", "GE"], 240).astype(object)
    scanner[0] = "Philips"

    row = screen(pd.DataFrame({"diagnosis": diagnosis, "scanner": scanner}))["scanner"]

    assert row["risk"] in ("NONE", "LOW")
    assert "1 categories with fewer than 10 rows pooled" in row["reason"]


def test_many_small_categories_cannot_vouch_for_themselves(diagnosis):
    """Every row's category estimate comes from other folds, so small sites do not look pure."""
    sites = np.repeat([f"site{i:02d}" for i in range(20)], 12)

    row = screen(pd.DataFrame({"diagnosis": diagnosis, "site": sites}))["site"]

    assert row["risk"] in ("NONE", "LOW"), row


def test_categories_that_are_all_too_small_are_not_scored(diagnosis):
    row = screen(pd.DataFrame({"diagnosis": diagnosis, "visit_code": [f"V{i}" for i in range(240)]}))["visit_code"]

    assert row["risk"] == "N/A" and "no category has 10 or more rows" in row["reason"]


def test_informative_missingness_is_flagged(diagnosis):
    biomarker_status = np.where(diagnosis == 1, None, "measured")

    row = screen(pd.DataFrame({"diagnosis": diagnosis, "biomarker_status": biomarker_status}))["biomarker_status"]

    assert row["risk"] == "EXTREME"


def test_numeric_codes_with_few_values_are_treated_as_categories(diagnosis):
    """Site codes 1-4 have no order; sites 1 and 3 are all patients, 2 and 4 all controls."""
    site = np.where(diagnosis == 1, np.random.default_rng(14).choice([1, 3], 240),
                    np.random.default_rng(15).choice([2, 4], 240))

    row = screen(pd.DataFrame({"diagnosis": diagnosis, "site": site}))["site"]

    assert (row["risk"], row["kind"]) == ("EXTREME", "categorical")


def test_string_multiclass_targets_are_screened():
    stage = np.tile(["CN", "MCI", "AD"], 40)
    clinic = np.where(stage == "AD", "memory clinic", "community")

    row = screen(pd.DataFrame({"stage": stage, "clinic": clinic}), target="stage")["clinic"]

    assert row["risk"] in ("HIGH", "EXTREME")


def test_a_continuous_target_is_not_screened():
    rng = np.random.default_rng(16)
    rows = confound_screen(pd.DataFrame({"age": rng.uniform(55, 90, 50), "sex": rng.choice(["F", "M"], 50)}), "age")

    assert all(r["risk"] == "N/A" and "target is continuous" in r["reason"] for r in rows)


def test_rows_without_a_target_are_ignored(diagnosis):
    frame = pd.DataFrame({"diagnosis": diagnosis.astype(float), "proxy": diagnosis * 1.0})
    frame.loc[:9, "diagnosis"] = np.nan

    assert screen(frame)["proxy"]["risk"] == "EXTREME"


def test_screening_is_reproducible(diagnosis):
    frame = pd.DataFrame({"diagnosis": diagnosis, "site": np.repeat(list("ABCDEFGH"), 30)})

    assert confound_screen(frame, "diagnosis", seed=1) == confound_screen(frame, "diagnosis", seed=1)


def longitudinal(n_subjects=40, visits=12, seed=0):
    """Several scans per subject, with a per-subject diagnosis."""
    rng = np.random.default_rng(seed)
    per_subject = rng.integers(0, 2, n_subjects)
    return pd.DataFrame({
        "scan_id": [f"S{i}" for i in range(n_subjects * visits)],
        "subject": np.repeat([f"P{i:03d}" for i in range(n_subjects)], visits),
        "diagnosis": np.repeat(per_subject, visits),
        "enrol_code": np.repeat([f"E{i}" for i in rng.permutation(n_subjects)], visits),  # unrelated to diagnosis
        "site": np.repeat(np.where(per_subject == 1, "memory clinic", "community"), visits),  # a real proxy
    })


def test_the_group_by_column_is_not_screened():
    rows = screen(longitudinal(), id_cols=["scan_id"], group_col="subject")

    assert rows["subject"]["risk"] == "N/A"
    assert rows["subject"]["reason"] == "Group By column (identifies subjects)"


def test_without_groups_a_subject_level_code_looks_like_a_proxy():
    """Why grouping matters: a subject's other visits vouch for its code."""
    rows = screen(longitudinal(), id_cols=["scan_id"])

    assert rows["enrol_code"]["risk"] == "EXTREME"
    assert rows["subject"]["risk"] == "EXTREME"


def test_with_groups_subject_level_codes_are_not_proxies():
    rows = screen(longitudinal(), id_cols=["scan_id"], group_col="subject")

    assert rows["enrol_code"]["risk"] in ("NONE", "LOW")


def test_with_groups_a_real_proxy_is_still_flagged():
    rows = screen(longitudinal(), id_cols=["scan_id"], group_col="subject")

    assert rows["site"]["risk"] == "EXTREME"


def test_too_few_groups_per_class_leaves_categories_unscored():
    frame = longitudinal(n_subjects=5)
    frame["diagnosis"] = np.where(frame["subject"] == "P000", 1, 0)  # only one subject has the diagnosis

    rows = screen(frame, id_cols=["scan_id"], group_col="subject")

    assert rows["site"]["reason"] == "too few groups per class to keep groups apart"


# ---------------------------------------------------------------------------
# Subgroup performance
# ---------------------------------------------------------------------------

def test_global_metrics():
    result = subgroup_performance(PROBS, LABELS, GROUPS, "group")
    g = result["global"]

    assert g["n"] == 6
    assert g["accuracy"] == pytest.approx(4 / 6)
    assert g["tpr"] == pytest.approx(2 / 3)
    assert g["fpr"] == pytest.approx(1 / 3)
    assert g["precision"] == pytest.approx(2 / 3)
    assert g["auc"] == pytest.approx(8 / 9)
    assert g["positive_rate"] == pytest.approx(0.5)


def test_per_group_metrics_and_fairness_gaps():
    result = subgroup_performance(PROBS, LABELS, GROUPS, "group", min_group_size=1)
    a, b = result["groups"]["A"], result["groups"]["B"]

    assert (a["accuracy"], a["tpr"], a["fpr"]) == pytest.approx((1.0, 1.0, 0.0))
    assert (b["accuracy"], b["tpr"], b["fpr"]) == pytest.approx((1 / 3, 0.0, 0.5))
    assert result["gaps"] == pytest.approx({
        "eo_tpr": 1.0, "eo_fpr": 0.5, "demographic_parity": 1 / 3, "equalized_odds": 1.0,
    })


def test_two_column_probabilities_match_one_column_scores():
    one_col = subgroup_performance(PROBS, LABELS, GROUPS, "group", min_group_size=1)
    two_col = subgroup_performance(np.column_stack([1 - PROBS, PROBS]), LABELS, GROUPS, "group", min_group_size=1)

    assert two_col["gaps"] == pytest.approx(one_col["gaps"])
    assert two_col["global"]["accuracy"] == pytest.approx(one_col["global"]["accuracy"])
    assert two_col["global"]["auc"] == pytest.approx(one_col["global"]["auc"])


def test_missing_group_values_are_excluded_from_slices():
    groups = pd.DataFrame({"group": ["A", "A", "A", None, "B", "B"]})

    result = subgroup_performance(PROBS, LABELS, groups, "group")

    assert set(result["groups"]) == {"A", "B"}
    assert result["groups"]["B"]["n"] == 2
    assert result["global"]["n"] == 6


def test_a_single_group_has_no_measurable_gap_and_one_class_slices_have_nan_auc():
    result = subgroup_performance(PROBS[:3], LABELS[:3], GROUPS.iloc[:3], "group", min_group_size=1)

    assert math.isnan(result["gaps"]["eo_tpr"])
    single_class = subgroup_performance(np.array([0.9, 0.8]), np.array([1, 1]),
                                        pd.DataFrame({"group": ["A", "A"]}), "group")
    assert math.isnan(single_class["global"]["auc"])


def test_subgroup_performance_validates_inputs():
    with pytest.raises(ValueError, match="slice_by"):
        subgroup_performance(PROBS, LABELS, GROUPS, "sex")
    with pytest.raises(ValueError, match="length"):
        subgroup_performance(PROBS[:5], LABELS[:5], GROUPS, "group")
    with pytest.raises(ValueError, match="groups column"):
        subgroup_performance(PROBS, LABELS, GROUPS, "group", groups="patient")


def test_a_rate_without_cases_is_nan_not_zero():
    result = subgroup_performance(PROBS, np.array([1, 0, 1, 0, 0, 0]), GROUPS, "group", min_group_size=1)

    assert math.isnan(result["groups"]["B"]["tpr"])  # B has no positives
    assert result["fairness"]["gaps"]["equal_opportunity"]["high"] is None


def test_subgroup_gaps_are_the_fairness_audit_point_estimates():
    result = subgroup_performance(PROBS, LABELS, GROUPS, "group", min_group_size=1)

    assert result["gaps"]["eo_fpr"] == result["fairness"]["gaps"]["predictive_equality"]["value"]
    assert result["groups"]["A"]["eligible"] and result["fairness"]["min_group_size"] == 1


def test_small_groups_are_marked_in_the_subgroup_view():
    result = subgroup_performance(PROBS, LABELS, GROUPS, "group")  # groups of 3, default minimum 20

    assert not result["groups"]["A"]["eligible"]
    assert math.isnan(result["gaps"]["eo_tpr"])


# ---------------------------------------------------------------------------
# Fairness audit: intervals, minimum group size, multiclass, regression
# ---------------------------------------------------------------------------

def outcomes(groups):
    """Labels, predictions and slices from {group: (n_correct, n_wrong)}; every row is a positive."""
    y_true, y_pred, slices = [], [], []
    for name, (right, wrong) in groups.items():
        y_true += [1] * (right + wrong)
        y_pred += [1] * right + [0] * wrong
        slices += [name] * (right + wrong)
    return np.array(y_true), np.array(y_pred), np.array(slices)


def binary(y_true, y_pred, slices, **kwargs):
    return fairness_audit(y_true, y_pred, slices, classes=[0, 1], **kwargs)


def test_hand_checked_binary_groups_and_gaps():
    preds = (PROBS >= 0.5).astype(int)

    result = binary(LABELS, preds, GROUPS["group"], min_group_size=1)

    assert result["task"] == "binary" and result["positive_class"] == 1
    a, b = result["groups"]["A"], result["groups"]["B"]
    assert a["confusion"] == [[1, 0], [0, 2]] and b["confusion"] == [[1, 1], [1, 0]]
    assert b["metrics"]["fpr"]["value"] == pytest.approx(0.5)
    assert b["metrics"]["selection_rate"]["value"] == pytest.approx(1 / 3)
    tpr_gap = result["gaps"]["equal_opportunity"]
    assert (tpr_gap["value"], tpr_gap["high"], tpr_gap["low"]) == (1.0, "A", "B")
    assert tpr_gap["ci"] == (1.0, 1.0)  # whenever both groups have a positive, A finds it and B misses it
    assert tpr_gap["flagged"]
    assert result["gaps"]["predictive_equality"]["high"] == "B"
    assert result["gaps"]["equalized_odds"]["metric"] == "tpr"  # the larger of the two components
    assert result["global"]["metrics"]["accuracy"]["value"] == pytest.approx(4 / 6)


def test_the_positive_class_can_be_either_class():
    preds = (PROBS >= 0.5).astype(int)

    as_one = binary(LABELS, preds, None, positive_class=1)["global"]["metrics"]
    as_zero = binary(LABELS, preds, None, positive_class=0)["global"]["metrics"]

    assert as_zero["tpr"]["value"] == pytest.approx(as_one["tnr"]["value"])
    assert as_zero["selection_rate"]["value"] == pytest.approx(1 - as_one["selection_rate"]["value"])


def test_an_interval_matches_the_normal_approximation_for_a_simple_rate():
    y_true, y_pred, slices = outcomes({"A": (80, 20)})

    accuracy = binary(y_true, y_pred, slices, n_boot=2000)["groups"]["A"]["metrics"]["accuracy"]

    half_width = 1.96 * math.sqrt(0.8 * 0.2 / 100)
    assert accuracy["value"] == pytest.approx(0.8)
    assert accuracy["ci"] == pytest.approx((0.8 - half_width, 0.8 + half_width), abs=0.02)


def test_a_large_gap_between_large_groups_is_flagged():
    y_true, y_pred, slices = outcomes({"A": (180, 20), "B": (100, 100)})

    gap = binary(y_true, y_pred, slices)["gaps"]["equal_opportunity"]

    assert gap["value"] == pytest.approx(0.4) and gap["ci"][0] > 0.3
    assert gap["flagged"] and not gap["noise"]


def test_a_large_gap_between_small_groups_is_marked_as_possible_noise():
    y_true, y_pred, slices = outcomes({"A": (12, 8), "B": (8, 12)})  # TPR 0.6 vs 0.4, 20 people each

    gap = binary(y_true, y_pred, slices)["gaps"]["equal_opportunity"]

    assert gap["value"] == pytest.approx(0.2) and gap["value"] > DISPARITY_THRESHOLD
    assert gap["ci"][0] < 0 < gap["ci"][1]
    assert gap["noise"] and not gap["flagged"]


def test_groups_with_no_real_difference_have_intervals_around_zero():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, 2000)
    y_pred = np.where(rng.random(2000) < 0.8, y_true, 1 - y_true)  # the same 80% accuracy for everyone

    result = binary(y_true, y_pred, rng.choice(["F", "M"], 2000))

    for gap in result["gaps"].values():
        assert gap["ci"][0] < 0 < gap["ci"][1] or abs(gap["value"]) < 0.05
        assert not gap["flagged"]


def test_groups_below_the_minimum_size_are_shown_but_not_compared():
    y_true, y_pred, slices = outcomes({"A": (20, 5), "B": (15, 10), "C": (0, 5)})

    result = binary(y_true, y_pred, slices)

    assert [result["groups"][g]["eligible"] for g in "ABC"] == [True, True, False]
    assert result["groups"]["C"]["metrics"]["tpr"]["value"] == 0.0  # still reported
    gap = result["gaps"]["equal_opportunity"]
    assert (gap["high"], gap["low"]) == ("A", "B") and gap["value"] == pytest.approx(0.2)


def test_without_two_large_enough_groups_there_is_no_gap():
    y_true, y_pred, slices = outcomes({"A": (20, 5), "B": (1, 1)})

    gap = binary(y_true, y_pred, slices)["gaps"]["equal_opportunity"]

    assert math.isnan(gap["value"]) and gap["ci"] is None and not gap["flagged"]
    assert f"at least {MIN_GROUP_SIZE} rows" in gap["reason"]


def repeated_scans(n_subjects=40, scans=10):
    """Each subject is scanned `scans` times and the model is right on all of them or none."""
    rng = np.random.default_rng(0)
    subject = np.repeat(np.arange(n_subjects), scans)
    correct = rng.random(n_subjects) < 0.7
    sex = np.where(np.arange(n_subjects) % 2, "F", "M")
    y_true = np.ones(n_subjects * scans, dtype=int)
    return y_true, correct[subject].astype(int), sex[subject], subject


def test_resampling_subjects_widens_intervals_for_repeated_scans():
    y_true, y_pred, sex, subject = repeated_scans()

    by_row = binary(y_true, y_pred, sex, min_group_size=1)
    by_subject = binary(y_true, y_pred, sex, groups=subject, min_group_size=1)

    def width(result):
        lo, hi = result["groups"]["F"]["metrics"]["accuracy"]["ci"]
        return hi - lo

    assert by_subject["bootstrap"]["unit"] == "group" and by_subject["bootstrap"]["units"] == 40
    assert by_subject["groups"]["F"]["units"] == 20 and by_row["groups"]["F"]["units"] == 200
    assert width(by_subject) > 2 * width(by_row)  # ten copies of a scan are one piece of evidence, not ten


def test_the_minimum_group_size_counts_subjects_when_groups_are_given():
    y_true, y_pred, sex, subject = repeated_scans()

    assert binary(y_true, y_pred, sex, min_group_size=25)["groups"]["F"]["eligible"]  # 200 scans
    assert not binary(y_true, y_pred, sex, groups=subject, min_group_size=25)["groups"]["F"]["eligible"]  # 20 people


def test_a_subject_in_two_slices_counts_in_both():
    result = binary([1, 1, 0, 0], [1, 0, 0, 0], ["young", "old", "young", "old"], groups=["P1", "P1", "P2", "P3"],
                    min_group_size=1)

    assert result["groups"]["old"]["units"] == 2 and result["groups"]["young"]["units"] == 2
    assert result["global"]["units"] == 3


def test_results_are_reproducible_and_depend_on_the_seed():
    y_true, y_pred, slices = outcomes({"A": (30, 10), "B": (20, 20)})

    first, again = binary(y_true, y_pred, slices, seed=3), binary(y_true, y_pred, slices, seed=3)
    other = binary(y_true, y_pred, slices, seed=4)

    assert repr(first) == repr(again)  # repr: NaN metrics compare equal
    assert other["gaps"]["equal_opportunity"]["ci"] != first["gaps"]["equal_opportunity"]["ci"]
    assert other["gaps"]["equal_opportunity"]["value"] == first["gaps"]["equal_opportunity"]["value"]


def test_the_bootstrap_does_not_depend_on_its_block_size(monkeypatch):
    import prism.analysis as analysis
    y_true, y_pred, slices = outcomes({"A": (30, 10), "B": (20, 20)})
    whole = binary(y_true, y_pred, slices, n_boot=50)
    monkeypatch.setattr(analysis, "BOOTSTRAP_BUDGET_BYTES", 24 * 80 * 7)  # 7 resamples per block

    assert repr(binary(y_true, y_pred, slices, n_boot=50)) == repr(whole)


def test_without_resamples_there_are_no_intervals():
    y_true, y_pred, slices = outcomes({"A": (180, 20), "B": (100, 100)})

    result = binary(y_true, y_pred, slices, n_boot=0)

    gap = result["gaps"]["equal_opportunity"]
    assert gap["ci"] is None and gap["flagged"]  # flagged on size alone, and said so
    assert result["groups"]["A"]["metrics"]["tpr"]["ci"] is None
    assert any("No resamples" in note for note in result["notes"])


def test_rows_without_a_slice_value_count_only_globally():
    result = binary([1, 1, 0, 0, 1], [1, 0, 0, 0, 1], ["A", "A", None, "B", "B"], min_group_size=1)

    assert {name: g["n"] for name, g in result["groups"].items()} == {"A": 2, "B": 2}
    assert result["global"]["n"] == 5
    assert any("1 rows have no value to slice by" in note for note in result["notes"])


def test_rows_without_a_group_value_are_resampled_alone():
    result = binary([1, 1, 0, 0], [1, 0, 0, 0], None, groups=["P1", "P1", None, None])

    assert result["bootstrap"]["units"] == 3
    assert any("no Group By value" in note for note in result["notes"])


def test_rows_with_unknown_labels_are_left_out():
    result = fairness_audit(["AD", "CN", None, "CN"], ["AD", "CN", "AD", "MCI"], None, classes=["AD", "CN"])

    assert result["global"]["n"] == 2
    assert any("2 rows" in note for note in result["notes"])


def test_many_groups_carry_a_selection_caveat():
    y_true, y_pred, slices = outcomes({"A": (20, 5), "B": (15, 10), "C": (18, 7)})

    assert any("largest of several differences" in note for note in binary(y_true, y_pred, slices)["notes"])


def test_multiclass_groups_and_gaps():
    # Group X: everything right. Group Y: every "b" called "a".
    y_true = ["a", "b", "c"] * 2 + ["a", "b", "c"] * 2
    y_pred = ["a", "b", "c"] * 2 + ["a", "a", "c"] * 2
    slices = ["X"] * 6 + ["Y"] * 6

    result = fairness_audit(y_true, y_pred, slices, classes=["a", "b", "c"], min_group_size=1)

    assert result["task"] == "multiclass" and result["positive_class"] is None
    y = result["groups"]["Y"]["metrics"]
    assert y["accuracy"]["value"] == pytest.approx(4 / 6)
    assert y["balanced_accuracy"]["value"] == pytest.approx(2 / 3)
    assert (y["recall:a"]["value"], y["recall:b"]["value"]) == (1.0, 0.0)
    assert y["selection_rate:a"]["value"] == pytest.approx(4 / 6)
    assert result["groups"]["Y"]["confusion"] == [[2, 0, 0], [2, 0, 0], [0, 0, 2]]
    gaps = result["gaps"]
    assert gaps["recall:b"]["value"] == 1.0 and (gaps["recall:b"]["high"], gaps["recall:b"]["low"]) == ("X", "Y")
    assert gaps["recall:a"]["value"] == 0.0
    assert gaps["selection_rate:a"]["value"] == pytest.approx(1 / 3)
    assert set(gaps) == {"accuracy", "balanced_accuracy", "recall:a", "recall:b", "recall:c",
                         "selection_rate:a", "selection_rate:b", "selection_rate:c"}


def test_multiclass_classes_default_to_the_sorted_labels():
    result = fairness_audit([2, 0, 1, 2], [2, 0, 1, 1], None)

    assert result["classes"] == [0, 1, 2] and result["task"] == "multiclass"


def test_regression_groups_match_direct_computation():
    rng = np.random.default_rng(0)
    truth = rng.normal(70, 8, 300)
    sex = np.array(["F", "M"] * 150)
    pred = truth + rng.normal(0, 2, 300) + np.where(sex == "F", 1.5, 0.0)  # women are over-estimated

    result = fairness_audit(truth, pred, sex, task="regression")

    f = result["groups"]["F"]["metrics"]
    err, t = (pred - truth)[sex == "F"], truth[sex == "F"]
    assert f["rmse"]["value"] == pytest.approx(np.sqrt(np.mean(err ** 2)))
    assert f["mae"]["value"] == pytest.approx(np.mean(np.abs(err)))
    assert f["mean_error"]["value"] == pytest.approx(np.mean(err))
    assert f["r2"]["value"] == pytest.approx(1 - np.sum(err ** 2) / np.sum((t - t.mean()) ** 2))
    bias = result["gaps"]["mean_error"]
    assert (bias["high"], bias["low"]) == ("F", "M") and bias["value"] == pytest.approx(1.5, abs=0.5)
    assert bias["flagged"]  # regression gaps are flagged when the interval excludes 0
    assert result["task"] == "regression" and result["classes"] is None


def test_regression_r2_is_nan_for_a_group_with_constant_truth():
    result = fairness_audit([5.0, 5.0, 1.0, 3.0], [5.5, 4.5, 1.0, 2.0], ["A", "A", "B", "B"], task="regression",
                            min_group_size=1)

    assert math.isnan(result["groups"]["A"]["metrics"]["r2"]["value"])
    assert result["groups"]["A"]["metrics"]["rmse"]["value"] == pytest.approx(0.5)


def test_regression_accepts_a_column_of_predictions():
    result = fairness_audit(np.arange(5.0), np.arange(5.0).reshape(5, 1) + 1, None, task="regression")

    assert result["global"]["metrics"]["mean_error"]["value"] == pytest.approx(1.0)


def test_fairness_audit_validates_inputs():
    with pytest.raises(ValueError, match="task"):
        fairness_audit([1, 0], [1, 0], task="ranking")
    with pytest.raises(ValueError, match="one prediction per row"):
        fairness_audit([1, 0], [1, 0, 1])
    with pytest.raises(ValueError, match="n_boot"):
        fairness_audit([1, 0], [1, 0], n_boot=-1)
    with pytest.raises(ValueError, match="positive class"):
        fairness_audit([1, 0], [1, 0], classes=[0, 1], positive_class=2)
    with pytest.raises(ValueError, match="slices has"):
        fairness_audit([1, 0], [1, 0], ["A"])
    with pytest.raises(ValueError, match="two classes"):
        fairness_audit([1, 1], [1, 1])


# ---------------------------------------------------------------------------
# Linear probe
# ---------------------------------------------------------------------------

def test_probe_finds_a_linearly_encoded_attribute():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    features = np.column_stack([y + rng.normal(0, 0.1, 200), rng.normal(size=200)])

    result = linear_probe(features, y)

    assert result["auc_mean"] > 0.99
    assert result["n_splits"] == 5 and len(result["folds"]) == 5
    assert result["n_samples"] == 200


def test_probe_on_noise_is_near_chance():
    rng = np.random.default_rng(1)

    result = linear_probe(rng.normal(size=(200, 5)), rng.integers(0, 2, 200))

    assert 0.3 < result["auc_mean"] < 0.7


def test_probe_is_deterministic_for_a_seed():
    rng = np.random.default_rng(2)
    features, y = rng.normal(size=(80, 3)), rng.integers(0, 2, 80)

    assert linear_probe(features, y, seed=7) == linear_probe(features, y, seed=7)


def test_probe_reduces_folds_for_rare_classes_and_flattens_3d_input():
    rng = np.random.default_rng(3)
    y = np.array([0] * 30 + [1] * 3)

    result = linear_probe(rng.normal(size=(33, 2, 2)), y)

    assert result["n_splits"] == 3


def test_probe_supports_multiclass_targets():
    rng = np.random.default_rng(4)
    y = np.repeat(["CN", "MCI", "AD"], 40)
    centres = {"CN": 0.0, "MCI": 3.0, "AD": 6.0}
    features = np.array([[centres[c]] for c in y]) + rng.normal(0, 0.2, (120, 1))

    assert linear_probe(features, y)["auc_mean"] > 0.99


def test_probe_reports_single_class_targets():
    result = linear_probe(np.zeros((10, 2)), np.ones(10))

    assert math.isnan(result["auc_mean"]) and "fewer than 2 classes" in result["error"]


def test_probe_can_hold_out_whole_subjects():
    """Several scans per subject: without grouped CV the probe memorises subject identity."""
    rng = np.random.default_rng(5)
    n_subjects, scans = 40, 5
    subject = np.repeat(np.arange(n_subjects), scans)
    subject_embedding = rng.normal(size=(n_subjects, 20))
    features = subject_embedding[subject] + rng.normal(0, 0.05, (n_subjects * scans, 20))
    label = rng.integers(0, 2, n_subjects)[subject]  # unrelated to the embedding

    result = linear_probe(features, label, groups=subject)

    assert result["auc_mean"] < 0.75


def subject_level_data(n_subjects=40, scans=5, dims=50, seed=5):
    """Features that identify the subject and nothing else; labels random per subject."""
    rng = np.random.default_rng(seed)
    subject = np.repeat(np.arange(n_subjects), scans)
    features = rng.normal(size=(n_subjects, dims))[subject] + rng.normal(0, 0.05, (n_subjects * scans, dims))
    label = rng.integers(0, 2, n_subjects)[subject]
    return features, label, subject


def test_without_groups_subject_identity_inflates_the_probe():
    """Why Group By matters: the same data looks highly 'encoded' when subjects leak across folds."""
    features, label, subject = subject_level_data()

    leaky = linear_probe(features, label)
    grouped = linear_probe(features, label, groups=subject)

    assert leaky["auc_mean"] > 0.9
    assert grouped["auc_mean"] < 0.75
    assert (grouped["grouped"], grouped["n_groups"]) == (True, 40)


def test_grouped_probe_needs_two_groups_per_class():
    features = np.zeros((6, 2))
    label = np.array([0, 0, 0, 1, 1, 1])

    result = linear_probe(features, label, groups=[1, 1, 2, 3, 3, 3])

    assert "Class '1' has too few groups (1)" in result["error"]


def test_balanced_accuracy_is_reported_with_its_chance_level():
    rng = np.random.default_rng(6)
    y = np.repeat(["a", "b", "c"], 30)
    features = np.array([[{"a": 0.0, "b": 3.0, "c": 6.0}[v]] for v in y]) + rng.normal(0, 0.2, (90, 1))

    result = linear_probe(features, y)

    assert result["balanced_accuracy_mean"] > 0.95
    assert result["chance_balanced_accuracy"] == pytest.approx(1 / 3)
    assert result["classes"] == ["a", "b", "c"]


def test_missing_targets_are_dropped_together_with_their_groups():
    features, label, subject = subject_level_data(n_subjects=20)
    target = label.astype(object)
    target[[0, 7]] = None

    result = linear_probe(features, target, groups=subject)

    assert result["error"] is None
    assert (result["n_samples"], result["dropped_missing"]) == (98, 2)
    assert "2 rows with a missing target were left out." in result["notes"]


def test_control_separates_a_real_signal_from_chance():
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, 120)
    encoded = np.column_stack([y + rng.normal(0, 0.3, 120), rng.normal(size=120)])
    noise = rng.normal(size=(120, 2))

    real = linear_probe(encoded, y, n_controls=20)["control"]
    chance = linear_probe(noise, y, n_controls=20)["control"]

    assert real["n"] == 20 and real["n_at_least_real"] == 0
    assert real["p_value"] == pytest.approx(1 / 21)
    assert 0.35 < real["mean"] < 0.65
    assert chance["p_value"] > 0.05


def test_control_shuffles_whole_subjects_when_labels_are_per_subject():
    features, label, subject = subject_level_data(n_subjects=20)

    result = linear_probe(features, label, groups=subject, n_controls=5)

    assert result["control"]["shuffled"] == "group"


def test_shuffling_between_groups_keeps_each_group_uniform():
    from prism.analysis import _shuffle_labels
    groups = np.repeat(np.arange(10), 3)
    y = np.repeat(np.arange(10) % 2, 3)

    shuffled = _shuffle_labels(y, groups, "group", np.random.default_rng(0))

    assert all(len(set(shuffled[groups == g])) == 1 for g in range(10))
    assert sorted(shuffled[::3]) == sorted(y[::3])


def test_labels_that_vary_within_groups_fall_back_to_row_shuffles():
    rng = np.random.default_rng(8)
    groups = np.repeat(np.arange(20), 4)
    y = rng.integers(0, 2, 80)  # e.g. diagnosis changing between visits

    result = linear_probe(rng.normal(size=(80, 2)), y, groups=groups, n_controls=3)

    assert result["control"]["shuffled"] == "row"
    assert any("varies within some groups" in note for note in result["notes"])


def test_numeric_targets_with_many_values_get_a_regression_probe():
    rng = np.random.default_rng(9)
    age = rng.uniform(55, 90, 100)
    features = np.column_stack([(age - 70) / 10 + rng.normal(0, 0.05, 100), rng.normal(size=100)])

    result = linear_probe(features, age, groups=np.repeat(np.arange(25), 4), n_controls=5)

    assert (result["task"], result["metric"]) == ("regression", "R²")
    assert result["r2_mean"] > 0.9 and result["score_mean"] == result["r2_mean"]
    assert result["control"]["mean"] < 0.2


@pytest.mark.parametrize("target, message", [
    (np.full(20, 3.5), "zero variance"),
    (np.arange(8, dtype=float) * 1.5, "fewer than 10"),
])
def test_regression_probe_reports_unusable_targets(target, message):
    result = linear_probe(np.zeros((len(target), 2)), target, task="regression")

    assert message in result["error"]


def test_controls_are_reproducible_for_a_seed():
    features, label, subject = subject_level_data(n_subjects=20)

    first = linear_probe(features, label, groups=subject, n_controls=4, seed=3)
    second = linear_probe(features, label, groups=subject, n_controls=4, seed=3)

    assert first["control"] == second["control"]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def test_format_audit_report_renders_all_sections(screening_frame):
    screening = confound_screen(screening_frame, "diagnosis", id_cols=["subject_id"])
    performance = subgroup_performance(PROBS, LABELS, GROUPS, "group", min_group_size=1)
    probes = {"layer4": {"auc_mean": 0.81, "auc_std": 0.02, "folds": [0.8, 0.82]}}

    report = format_audit_report("Demo audit", screening, performance, probes, thesis_ref="Table 4.2")

    assert report.startswith("# Demo audit")
    assert "_Maps to: Table 4.2_" in report
    assert "| proxy | **EXTREME** | 1.000 |" in report
    assert "| subject_id | **N/A** | — | identifier |" in report
    assert "### Sliced by `group`" in report
    assert "| B | 3 | 0.333 |" in report
    assert "ΔEO-TPR = 1.000 [1.000, 1.000] ⚠" in report
    assert "Intervals: 95% bootstrap, 1000 resamples of 6 rows." in report
    assert "| `layer4` | AUC | 0.810 ± 0.020 | 0.800, 0.820 | — |" in report


def test_format_audit_report_marks_small_groups_and_unmeasurable_gaps(screening_frame):
    screening = confound_screen(screening_frame, "diagnosis", id_cols=["subject_id"])
    performance = subgroup_performance(PROBS, LABELS, GROUPS, "group")  # groups of 3 < 20

    report = format_audit_report("Demo audit", screening, performance, {})

    assert "| B* | 3 | 0.333 |" in report
    assert "Below the minimum group size (20)" in report
    assert "ΔEO-TPR = n/a" in report



def test_the_bootstrap_reports_progress_and_can_be_stopped():
    from prism.core.cancel import Cancelled
    y_true, y_pred, slices = outcomes({"A": (30, 10), "B": (20, 20)})
    seen = []
    import prism.analysis as analysis
    budget = analysis.BOOTSTRAP_BUDGET_BYTES
    try:
        analysis.BOOTSTRAP_BUDGET_BYTES = 24 * 80 * 100  # 100 resamples per block
        binary(y_true, y_pred, slices, n_boot=500, progress=seen.append)

        def stop(fraction):
            raise Cancelled()

        with pytest.raises(Cancelled):
            binary(y_true, y_pred, slices, n_boot=500, progress=stop)
    finally:
        analysis.BOOTSTRAP_BUDGET_BYTES = budget

    assert seen == [0.2, 0.4, 0.6, 0.8, 1.0]
