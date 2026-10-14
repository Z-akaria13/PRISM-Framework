"""analysis.linear_probe's classifiers: shrinkage LDA and ridge (streamed, exact) and logistic (in memory)."""
import tracemalloc

import h5py
import numpy as np
import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from prism.analysis import LDA_SHRINKAGE, _macro_auc, _ovr_margins, linear_probe


def weak_signal(n=600, d=12, n_classes=2, seed=0, positive_rate=None):
    rng = np.random.default_rng(seed)
    if positive_rate is None:
        y = rng.integers(0, n_classes, n)
    else:
        y = (rng.random(n) < positive_rate).astype(int)
    X = rng.normal(size=(n, d))
    X[:, :3] += 0.6 * y[:, None]
    return X, y


def sklearn_macro_auc(y_true, scores, n_classes):
    """The reference AUC, computed with scikit-learn's own roc_auc_score."""
    if n_classes == 2:
        return roc_auc_score(y_true, scores[:, 1])
    return np.mean([roc_auc_score(y_true == k, scores[:, k]) for k in np.unique(y_true)])


def sklearn_ridge_aucs(X, y, splits, n_classes):
    """Per-fold AUC of scikit-learn's in-memory StandardScaler + RidgeClassifier
    (multiclass scores turned into one-vs-rest margins, as the probe does)."""
    aucs = []
    for train, test in splits:
        scores = make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0)).fit(X[train], y[train]).decision_function(X[test])
        scores = np.column_stack([-scores, scores]) if n_classes == 2 else _ovr_margins(scores)
        aucs.append(sklearn_macro_auc(y[test], scores, n_classes))
    return aucs


# ---------------------------------------------------------------------------
# Streamed ridge is exact
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_classes", [2, 3])
def test_streamed_ridge_matches_scikit_learn_in_memory(n_classes):
    X, y = weak_signal(n_classes=n_classes)
    splits = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))

    result = linear_probe(X, y, model="ridge", chunk_rows=37)

    np.testing.assert_allclose(result["folds"], sklearn_ridge_aucs(X, y, splits, n_classes), rtol=1e-9)
    assert result["model"] == "ridge"


def test_streamed_ridge_matches_scikit_learn_with_grouped_folds():
    X, y = weak_signal(n=400)
    groups = np.repeat(np.arange(100), 4)
    y = np.repeat(y[::4], 4)  # one label per subject
    splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X, y, groups))

    result = linear_probe(X, y, groups=groups, model="ridge", chunk_rows=50)

    np.testing.assert_allclose(result["folds"], sklearn_ridge_aucs(X, y, splits, 2), rtol=1e-9)


def test_streamed_regression_matches_scikit_learn_in_memory():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 8))
    age = 70 + 5 * X[:, 0] + rng.normal(0, 2, 300)
    splits = list(KFold(5, shuffle=True, random_state=42).split(X))
    expected = [r2_score(age[test], make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X[train], age[train]).predict(X[test]))
                for train, test in splits]

    result = linear_probe(X, age, chunk_rows=23)

    assert (result["task"], result["model"]) == ("regression", "ridge")
    np.testing.assert_allclose(result["folds"], expected, rtol=1e-9)


@pytest.mark.parametrize("task_data", ["classification", "regression"])
def test_chunk_size_does_not_change_the_result(task_data):
    X, y = weak_signal(n=500)
    target = y if task_data == "classification" else X[:, 0] * 3 + np.random.default_rng(2).normal(size=500)

    small = linear_probe(X, target, model="ridge", n_controls=3, chunk_rows=7)
    whole = linear_probe(X, target, model="ridge", n_controls=3, chunk_rows=10_000)

    np.testing.assert_allclose(small["folds"], whole["folds"], rtol=1e-9)
    np.testing.assert_allclose(small["control"]["scores"], whole["control"]["scores"], rtol=1e-9)


def test_constant_features_are_handled_like_standard_scaler():
    X, y = weak_signal()
    X[:, 5] = 3.0
    splits = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))

    result = linear_probe(X, y, model="ridge")

    np.testing.assert_allclose(result["folds"], sklearn_ridge_aucs(X, y, splits, 2), rtol=1e-9)


# ---------------------------------------------------------------------------
# Streaming from disk
# ---------------------------------------------------------------------------

def test_streamed_models_read_the_layer_from_disk_without_loading_it(tmp_path):
    n, d = 40_000, 256
    X, y = weak_signal(n=n, d=d)
    path = tmp_path / "layer.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("layer", data=X.astype(np.float32), chunks=(500, d))
    layer_bytes = n * d * 8  # the layer as float64, as the probes compute

    peaks = {}
    with h5py.File(path, "r") as f:
        for model in ("lda", "ridge", "logistic"):
            tracemalloc.start()
            linear_probe(f["layer"], y, model=model, chunk_rows=500)
            peaks[model] = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()

    assert peaks["logistic"] > layer_bytes  # loads it
    # Streamed: a chunk, the D x D totals and per-row bookkeeping, never the layer itself.
    assert peaks["lda"] < layer_bytes / 3
    assert peaks["ridge"] < layer_bytes / 3


def test_ridge_refuses_layers_whose_running_totals_would_not_fit():
    result = linear_probe(np.zeros((20, 30_000), dtype=np.float32), [0, 1] * 10, model="ridge")

    assert "would need" in result["error"] and "30000 features" in result["error"]


# ---------------------------------------------------------------------------
# Ridge behaves like the logistic probe
# ---------------------------------------------------------------------------

def test_ridge_and_logistic_agree_on_auc():
    X, y = weak_signal(n=2000, d=20)

    ridge, logistic = (linear_probe(X, y, model=m)["auc_mean"] for m in ("ridge", "logistic"))

    assert abs(ridge - logistic) < 0.02


def test_ridge_balanced_accuracy_is_fair_on_imbalanced_targets():
    X, y = weak_signal(n=3000, positive_rate=0.1, seed=3)

    ridge, logistic = (linear_probe(X, y, model=m) for m in ("ridge", "logistic"))

    assert ridge["balanced_accuracy_mean"] > 0.6
    assert abs(ridge["balanced_accuracy_mean"] - logistic["balanced_accuracy_mean"]) < 0.05


def test_ridge_finds_a_class_lying_between_two_others():
    """CN / MCI / AD along one severity axis: a straight one-vs-rest cut cannot isolate MCI."""
    rng = np.random.default_rng(4)
    y = np.repeat([0, 1, 2], 100)
    X = np.column_stack([y * 3.0, rng.normal(size=300)]) + rng.normal(0, 0.3, (300, 2))

    result = linear_probe(X, y, model="ridge")

    assert result["auc_mean"] > 0.99 and result["balanced_accuracy_mean"] > 0.9


def test_ridge_controls_separate_signal_from_chance():
    X, y = weak_signal(n=600)
    noise = np.random.default_rng(5).normal(size=X.shape)

    real = linear_probe(X, y, model="ridge", n_controls=20)["control"]
    chance = linear_probe(noise, y, model="ridge", n_controls=20)["control"]

    assert real["n_at_least_real"] == 0 and real["p_value"] == pytest.approx(1 / 21)
    assert chance["p_value"] > 0.05


def test_ridge_controls_shuffle_whole_subjects():
    X, y = weak_signal(n=400)
    groups = np.repeat(np.arange(100), 4)
    y = np.repeat(y[::4], 4)

    result = linear_probe(X, y, groups=groups, model="ridge", n_controls=5)

    assert result["control"]["shuffled"] == "group"


def test_controls_use_the_same_folds_as_the_real_probe():
    """Both models now share folds between the real probe and its controls."""
    X, y = weak_signal(n=300)

    logistic = linear_probe(X, y, model="logistic", n_controls=2, seed=9)
    again = linear_probe(X, y, model="logistic", n_controls=2, seed=9)

    assert logistic["control"] == again["control"]


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="Unknown probe model"):
        linear_probe(np.zeros((10, 2)), [0, 1] * 5, model="svm")


def test_ovr_margins_compare_each_class_with_the_best_other():
    scores = np.array([[3.0, 1.0, 2.0], [0.0, 5.0, 4.0]])

    np.testing.assert_allclose(_ovr_margins(scores), [[1.0, -2.0, -1.0], [-5.0, 1.0, -1.0]])


# ---------------------------------------------------------------------------
# Shrinkage LDA (the default)
# ---------------------------------------------------------------------------

def sklearn_lda_aucs(X, y, splits, n_classes):
    """Per-fold AUC of scikit-learn's in-memory StandardScaler + shrinkage LDA."""
    aucs = []
    for train, test in splits:
        model = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage=LDA_SHRINKAGE))
        model.fit(X[train], y[train])
        if n_classes == 2:
            s = model.decision_function(X[test])
            scores = np.column_stack([-s, s])
        else:
            scores = model.predict_proba(X[test])
        aucs.append(sklearn_macro_auc(y[test], scores, n_classes))
    return aucs


@pytest.mark.parametrize("n_classes", [2, 3, 5])
def test_streamed_lda_matches_scikit_learn_in_memory(n_classes):
    X, y = weak_signal(n_classes=n_classes, n=750)
    splits = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))

    result = linear_probe(X, y, model="lda", chunk_rows=41)

    np.testing.assert_allclose(result["folds"], sklearn_lda_aucs(X, y, splits, n_classes), rtol=1e-9)


def test_streamed_lda_matches_scikit_learn_with_grouped_folds():
    X, y = weak_signal(n=400)
    groups = np.repeat(np.arange(100), 4)
    y = np.repeat(y[::4], 4)
    splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X, y, groups))

    result = linear_probe(X, y, groups=groups, model="lda", chunk_rows=50)

    np.testing.assert_allclose(result["folds"], sklearn_lda_aucs(X, y, splits, 2), rtol=1e-9)


def test_lda_is_the_default_model():
    X, y = weak_signal()

    assert linear_probe(X, y)["model"] == "lda"


def ordered_levels(seed=0):
    """Five ordered levels (very low ... very high) along one direction, plus noise features."""
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(5), 120)
    X = rng.normal(size=(600, 11))
    X[:, 0] += 1.0 * y
    return X, y


def test_lda_handles_ordered_levels_as_well_as_logistic():
    X, y = ordered_levels()

    lda, logistic, ridge = (linear_probe(X, y, model=m)["auc_mean"] for m in ("lda", "logistic", "ridge"))

    assert abs(lda - logistic) < 0.01
    assert ridge < lda - 0.02  # the masking that makes ridge the weaker multiclass choice


def test_lda_balanced_accuracy_is_fair_on_imbalanced_targets():
    X, y = weak_signal(n=3000, positive_rate=0.1, seed=3)

    lda, logistic = (linear_probe(X, y, model=m) for m in ("lda", "logistic"))

    assert lda["balanced_accuracy_mean"] > 0.6
    assert abs(lda["balanced_accuracy_mean"] - logistic["balanced_accuracy_mean"]) < 0.05


def test_lda_controls_separate_signal_from_chance():
    X, y = weak_signal(n=600)
    noise = np.random.default_rng(5).normal(size=X.shape)

    real = linear_probe(X, y, model="lda", n_controls=20)["control"]
    chance = linear_probe(noise, y, model="lda", n_controls=20)["control"]

    assert real["n_at_least_real"] == 0 and real["p_value"] == pytest.approx(1 / 21)
    assert chance["p_value"] > 0.05


def test_lda_chunk_size_does_not_change_the_result():
    X, y = ordered_levels()

    small = linear_probe(X, y, model="lda", n_controls=3, chunk_rows=7)
    whole = linear_probe(X, y, model="lda", n_controls=3, chunk_rows=10_000)

    np.testing.assert_allclose(small["folds"], whole["folds"], rtol=1e-9)
    np.testing.assert_allclose(small["control"]["scores"], whole["control"]["scores"], rtol=1e-9)


# ---------------------------------------------------------------------------
# Many controls: same answers, bounded memory
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_classes", [2, 3])
def test_woodbury_lda_matches_the_direct_solve(monkeypatch, n_classes):
    import prism.analysis as analysis
    X, y = weak_signal(n_classes=n_classes, n=450)
    splits = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))

    monkeypatch.setattr(analysis, "WOODBURY_MIN_SETS", 10**9)  # always solve each set directly
    direct = linear_probe(X, y, model="lda", n_controls=6)
    monkeypatch.setattr(analysis, "WOODBURY_MIN_SETS", 1)      # always factorise once and update
    woodbury = linear_probe(X, y, model="lda", n_controls=6)

    np.testing.assert_allclose(woodbury["folds"], direct["folds"], rtol=1e-9)
    np.testing.assert_allclose(woodbury["control"]["scores"], direct["control"]["scores"], rtol=1e-9)
    np.testing.assert_allclose(woodbury["folds"], sklearn_lda_aucs(X, y, splits, n_classes), rtol=1e-9)


@pytest.mark.parametrize("model, target_kind", [("lda", "classes"), ("ridge", "classes"), ("ridge", "numeric")])
def test_scoring_controls_in_small_batches_changes_nothing(monkeypatch, model, target_kind):
    import prism.analysis as analysis
    X, y = weak_signal(n=300)
    target = y if target_kind == "classes" else X[:, 0] * 2 + np.random.default_rng(6).normal(size=300)

    whole = linear_probe(X, target, model=model, n_controls=12)
    monkeypatch.setattr(analysis, "SCORE_BUDGET_BYTES", 1)  # one label set per pass over the data
    batched = linear_probe(X, target, model=model, n_controls=12)

    np.testing.assert_allclose(batched["folds"], whole["folds"], rtol=1e-12)
    np.testing.assert_allclose(batched["control"]["scores"], whole["control"]["scores"], rtol=1e-12)


def test_the_default_and_maximum_number_of_controls():
    from prism.analysis import DEFAULT_CONTROLS, MAX_CONTROLS

    assert (DEFAULT_CONTROLS, MAX_CONTROLS) == (50, 1000)


@pytest.mark.parametrize("model", ["lda", "ridge"])
def test_a_thousand_controls_run_and_resolve_p_to_one_in_a_thousand_and_one(model):
    X, y = weak_signal(n=300)

    control = linear_probe(X, y, model=model, n_controls=1000)["control"]

    assert control["n"] == 1000 and len(control["scores"]) == 1000
    assert control["p_value"] == pytest.approx(1 / 1001)


@pytest.mark.parametrize("bad", [-1, 1001, 2.5, True, "50"])
def test_out_of_range_control_counts_are_rejected(bad):
    with pytest.raises(ValueError, match="n_controls must be a whole number from 0 to 1000"):
        linear_probe(np.zeros((10, 2)), [0, 1] * 5, n_controls=bad)


@pytest.mark.parametrize("n_controls", [0, 20])
def test_lda_survives_classes_split_perfectly_by_a_constant_feature(n_controls):
    """Within-class variance is exactly zero. scikit-learn's LDA collapses to a zero solution here
    (AUC 0.5); PRISM's diagonal floor keeps the covariance invertible, so the perfect split is found."""
    y = np.tile([0, 1], 30)
    X = (y * 2.0)[:, None]

    result = linear_probe(X, y, model="lda", n_controls=n_controls)

    assert result["error"] is None and result["auc_mean"] == pytest.approx(1.0)



def test_rank_auc_matches_scikit_learn_including_ties():
    from prism.analysis import _auc
    rng = np.random.default_rng(7)
    labels = rng.integers(0, 2, 500).astype(bool)
    scores = np.round(rng.normal(size=500) + labels, 1)  # rounding creates many tied scores

    assert _auc(labels, scores) == pytest.approx(roc_auc_score(labels, scores), abs=1e-12)



# ---------------------------------------------------------------------------
# Progress and cancellation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model", ["lda", "ridge", "logistic"])
def test_the_probe_reports_rising_progress_up_to_one(model):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 8))
    seen = []

    linear_probe(X, X[:, 0] > 0, n_controls=3, model=model, chunk_rows=50, progress=seen.append)

    assert len(seen) > 10 and seen == sorted(seen) and seen[-1] == 1.0 and 0 < seen[0] < 0.2


def test_a_regression_probe_reports_progress():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    seen = []

    linear_probe(X, X[:, 0] * 10 + rng.normal(size=300), chunk_rows=50, progress=seen.append)

    assert len(seen) > 5 and seen == sorted(seen) and seen[-1] == 1.0


def test_raising_from_progress_stops_the_probe():
    from prism.core.cancel import Cancelled
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 8))
    calls = []

    def stop_at_third(fraction):
        calls.append(fraction)
        if len(calls) == 3:
            raise Cancelled()

    with pytest.raises(Cancelled):
        linear_probe(X, X[:, 0] > 0, n_controls=50, model="logistic", chunk_rows=50, progress=stop_at_third)
    assert len(calls) == 3  # nothing ran after the stop


# ---------------------------------------------------------------------------
# Fold AUCs for many label sets at once
# ---------------------------------------------------------------------------

def per_set_fold_aucs(label_sets, scores, splits, n_classes, model):
    """The one-set-at-a-time computation the batched _fold_aucs replaced, built on _macro_auc."""
    n_cols = 1 if n_classes == 2 else n_classes
    out = []
    for p, y in enumerate(label_sets):
        s = scores[:, p * n_cols:(p + 1) * n_cols]
        if n_classes == 2:
            s = np.column_stack([-s[:, 0], s[:, 0]])
        elif model == "lda":
            shifted = s - s.max(axis=1, keepdims=True)
            s = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
        else:
            s = _ovr_margins(s)
        out.append([_macro_auc(y[test], s[test], n_classes) for _, test in splits])
    return np.array(out)


@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("model", ["lda", "ridge"])
def test_batched_fold_aucs_match_the_per_set_computation(n_classes, model):
    from prism.analysis import _fold_aucs
    rng = np.random.default_rng(1)
    n_rows, n_sets, n_cols = 40, 12, (1 if n_classes == 2 else n_classes)
    splits = [(np.setdiff1d(np.arange(n_rows), test), test) for test in np.array_split(np.arange(n_rows), 4)]
    label_sets = rng.integers(0, n_classes, (n_sets, n_rows))
    scores = np.round(rng.normal(size=(n_rows, n_sets * n_cols)), 1)  # rounded: plenty of tied scores
    label_sets[1, splits[0][1]] = 0                                    # one class only in a fold
    scores[splits[1][1], 2 * n_cols:3 * n_cols] = np.nan              # a set's model undefined in a fold
    if n_classes == 3:
        label_sets[3, splits[2][1]] = rng.integers(0, 2, len(splits[2][1]))  # class 2 absent from a fold
    scores[:, 4 * n_cols:5 * n_cols] = 0.0                            # all scores tied

    batched = _fold_aucs(label_sets, scores, splits, n_classes, model)

    assert batched == pytest.approx(per_set_fold_aucs(label_sets, scores, splits, n_classes, model), nan_ok=True)
    assert np.isnan(batched[1, 0]) and np.isnan(batched[2, 1])
    assert np.nanmax(np.abs(batched[4] - 0.5)) < 1e-12  # tied scores give AUC 0.5
