"""Headless analysis primitives for PRISM audits.

These functions are the analysis behind the GUI's Evidence, Performance and
Probing tabs, with no Tk dependency, so they can also run headless on a
results cache. The Evidence tab calls `confound_screen`, Performance calls
`fairness_audit` and Probing calls `linear_probe`.

Three stages, one helper:

  - confound_screen(metadata, target_col, id_cols)
      Scores how well each metadata field alone predicts the target (a proxy,
      confound or leak). Returns a list of dicts: {field, risk, auc, kind, reason}.

  - fairness_audit(y_true, y_pred, slices, groups=None)
      Per-group performance (binary, multiclass or regression) and the
      fairness gaps between groups, each with a bootstrap confidence interval
      that resamples whole subjects when `groups` is given. Small groups are
      reported but left out of the gaps.

  - subgroup_performance(probs, labels, metadata, slice_by, positive_class)
      The binary view from model scores: accuracy, precision, recall, F1, AUC
      globally and per subgroup, with the gaps of `fairness_audit`.

  - linear_probe(features, target, groups=None, n_controls=0)
      Cross-validated linear probe: logistic regression (ROC-AUC, balanced
      accuracy) or ridge regression (R^2). `groups` (e.g. subject IDs) keeps
      each group in one fold; `n_controls` adds shuffled-label controls.

  - format_audit_report(...)
      Stitches the three outputs into a Markdown summary.
"""

from __future__ import annotations

import math
import warnings
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Stage 1: confound + leakage screening
# ---------------------------------------------------------------------------

_RISK_BANDS = [
    (0.95, "EXTREME"),
    (0.80, "HIGH"),
    (0.65, "MEDIUM"),
    (0.55, "LOW"),
]

MIN_CATEGORY_SIZE = 10     # smaller categories are pooled and cannot be flagged on their own
CATEGORICAL_MAX_UNIQUE = 10  # numeric columns with this many distinct values or fewer are categories


def _band_for(auc: float) -> str:
    for thresh, label in _RISK_BANDS:
        if auc >= thresh:
            return label
    return "NONE"


def confound_screen(
    metadata: pd.DataFrame,
    target_col: str,
    id_cols: Iterable[str] = (),
    group_col: str | None = None,
    seed: int = 42,
) -> list[dict]:
    """Score every other column for how well it alone predicts the (categorical) target.

    A high score means the column could stand in for the label, either as a confound the
    model may learn instead of the pathology (scanner, site) or as a leak (e.g. MMSE for
    an Alzheimer's diagnosis). Each row is {field, risk, auc, kind, reason}; `auc` is
    the strength of the association (0.5 = none, 1 = perfect, one-vs-rest macro average
    for >2 classes) and `risk` its band: EXTREME, HIGH, MEDIUM, LOW, NONE, or N/A when
    the column was not scored.

    * Continuous columns (numeric, more than 10 distinct values) are scored on their raw
      values, in whichever direction is stronger. Missing values are ignored.
    * Categorical columns are scored out-of-fold: each row's category->class frequencies
      come from the other folds, so a small category cannot vouch for itself. Categories
      with fewer than 10 rows are pooled into one "(rare)" bucket, and missing values form
      a "(missing)" category, since missingness can itself be informative.
    * `group_col` (e.g. the subject column chosen as Group By) is not screened, and the
      out-of-fold scoring keeps each group's rows in one fold. Otherwise, with several scans
      per subject, a subject's other scans vouch for it and identifier-like columns look like
      perfect proxies for a per-subject label.
    """
    if target_col not in metadata.columns:
        raise ValueError(f"target column '{target_col}' not in metadata")

    labelled = metadata[metadata[target_col].notna()].reset_index(drop=True)
    classes, y = np.unique(labelled[target_col].astype(str), return_inverse=True)
    if len(classes) < 2:
        return [{"field": target_col, "risk": "N/A", "auc": None, "kind": None,
                 "reason": "target has fewer than 2 classes"}]
    if _is_continuous(labelled[target_col]):
        return [{"field": col, "risk": "N/A", "auc": None, "kind": None,
                 "reason": "the target is continuous; screening needs a categorical target"}
                for col in metadata.columns]

    groups = None
    if group_col is not None:
        g = labelled[group_col].astype(object)
        groups = pd.factorize(g.where(g.notna(), [f"__row_{i}" for i in range(len(g))]))[0]

    id_set = set(id_cols)
    out: list[dict] = []
    for col in metadata.columns:
        if col == target_col:
            out.append(_unscored(col, "target column"))
        elif col in id_set:
            out.append(_unscored(col, "identifier"))
        elif col == group_col:
            out.append(_unscored(col, "Group By column (identifies subjects)"))
        elif _is_continuous(labelled[col]):
            out.append(_screen_continuous(col, labelled[col], y, len(classes)))
        else:
            out.append(_screen_categorical(col, labelled[col], y, len(classes), seed, groups))
    return out


def _is_continuous(series: pd.Series) -> bool:
    return (pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
            and series.nunique() > CATEGORICAL_MAX_UNIQUE)


def _unscored(col, reason, kind=None):
    return {"field": col, "risk": "N/A", "auc": None, "kind": kind, "reason": reason}


def _scored(col, auc, kind, reason):
    return {"field": col, "risk": _band_for(auc), "auc": round(float(auc), 3), "kind": kind, "reason": reason}


def _screen_continuous(col, values, y, n_classes):
    present = values.notna().to_numpy()
    x, y = values.to_numpy(dtype=float)[present], y[present]
    if len(x) < MIN_CATEGORY_SIZE:
        return _unscored(col, f"fewer than {MIN_CATEGORY_SIZE} rows with a value", "continuous")
    if len(np.unique(y)) < 2:
        return _unscored(col, "every row with a value has the same target", "continuous")
    targets = [y] if n_classes == 2 else [(y == k).astype(int) for k in np.unique(y)]
    strengths = []
    for t in targets:
        auc = roc_auc_score(t, x)
        strengths.append(max(auc, 1.0 - auc))  # a low value predicting the class counts too
    auc = float(np.mean(strengths))
    reason = f"value AUC = {auc:.3f}"
    if (~present).any():
        reason += f"; {int((~present).sum())} missing rows ignored"
    return _scored(col, auc, "continuous", reason)


def _screen_categorical(col, values, y, n_classes, seed, groups=None):
    labels = values.astype(object).where(values.notna(), "(missing)").astype(str)
    counts = labels.value_counts()
    rare = counts[counts < MIN_CATEGORY_SIZE].index
    if len(rare):
        labels = labels.where(~labels.isin(rare), "(rare)")
    codes = pd.factorize(labels)[0]
    if len(np.unique(codes)) < 2:
        reason = "single value" if len(rare) == 0 else f"no category has {MIN_CATEGORY_SIZE} or more rows"
        return _unscored(col, reason, "categorical")

    oof = _out_of_fold_class_rates(codes, y, n_classes, seed, groups)
    if oof is None:
        return _unscored(col, "too few groups per class to keep groups apart", "categorical")
    auc = _macro_auc(y, oof, n_classes)
    if np.isnan(auc):
        return _unscored(col, "could not be scored", "categorical")
    auc = max(auc, 0.5)  # out-of-fold scores below 0.5 are noise, not an inverse signal
    reason = f"cross-validated AUC = {auc:.3f}"
    if len(rare):
        reason += f"; {len(rare)} categories with fewer than {MIN_CATEGORY_SIZE} rows pooled"
    return _scored(col, auc, "categorical", reason)


def _out_of_fold_class_rates(codes, y, n_classes, seed, groups=None, smoothing=1.0):
    """For each row, its category's class frequencies estimated on the other folds (with
    `groups`, folds that keep each group whole). None if the groups are too few to split."""
    if groups is None:
        n_splits = int(min(5, max(2, np.bincount(y).min())))
        folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    else:
        n_splits = int(min(5, min(len(np.unique(groups[y == k])) for k in range(n_classes))))
        if n_splits < 2:
            return None
        folds = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rates = np.zeros((len(y), n_classes))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # "least populated class has only 1 member"
        splits = list(folds.split(codes, y, groups))
    for train, test in splits:
        prior = np.bincount(y[train], minlength=n_classes) / len(train)
        counts = np.zeros((codes.max() + 1, n_classes))
        np.add.at(counts, (codes[train], y[train]), 1)
        smoothed = (counts + smoothing * prior) / (counts.sum(axis=1, keepdims=True) + smoothing)
        rates[test] = smoothed[codes[test]]
    return rates


# ---------------------------------------------------------------------------
# Stage 2: subgroup performance
# ---------------------------------------------------------------------------

def _safe(metric_fn, *args, **kwargs):
    try:
        return float(metric_fn(*args, **kwargs))
    except Exception:  # noqa: BLE001
        return float("nan")


def _metrics_block(probs: np.ndarray, labels: np.ndarray,
                   positive_class: int) -> dict:
    if probs.ndim == 1:
        scores = probs
        preds = (probs >= 0.5).astype(int)
    else:
        scores = probs[:, positive_class]
        preds = probs.argmax(axis=1)

    n = len(labels)
    n_pos = int((labels == positive_class).sum())
    n_neg = n - n_pos

    # A rate with no cases to measure it on (e.g. TPR in a group without positives) is NaN, not 0.
    tpr = float(((preds == positive_class) & (labels == positive_class)).sum() / n_pos) if n_pos else float("nan")
    fpr = float(((preds == positive_class) & (labels != positive_class)).sum() / n_neg) if n_neg else float("nan")

    return {
        "n": n,
        "accuracy": _safe(accuracy_score, labels, preds),
        "precision": _safe(precision_score, labels, preds,
                           pos_label=positive_class, zero_division=0),
        "recall": _safe(recall_score, labels, preds,
                        pos_label=positive_class, zero_division=0),
        "f1": _safe(f1_score, labels, preds,
                    pos_label=positive_class, zero_division=0),
        "auc": _safe(roc_auc_score, labels, scores) if n_pos and n_neg
                else float("nan"),
        "tpr": tpr,
        "fpr": fpr,
        "positive_rate": float((preds == positive_class).mean()),
    }


def subgroup_performance(
    probs: np.ndarray,
    labels: np.ndarray,
    metadata: pd.DataFrame,
    slice_by: str,
    positive_class: int = 1,
    groups: str | None = None,
    n_boot: int = None,
    min_group_size: int = None,
    seed: int = 42,
) -> dict:
    """Binary metrics (one class against the rest) from model scores, globally and per subgroup
    of `metadata[slice_by]`, with the fairness gaps of `fairness_audit`. Scores are thresholded
    at 0.5 (one column) or by argmax (one column per class, labels 0..k-1).

    `groups` names a metadata column of subject IDs: the bootstrap then resamples subjects.
    Returns {slice_by, global, groups, gaps, fairness}: `gaps` holds the point estimates
    (eo_tpr, eo_fpr, demographic_parity, equalized_odds; NaN when fewer than two groups are
    large enough), `fairness` the full `fairness_audit` result with intervals and flags.
    """
    probs = np.asarray(probs)
    labels = np.asarray(labels)

    if slice_by not in metadata.columns:
        raise ValueError(f"slice_by column '{slice_by}' not in metadata")
    if len(metadata) != len(labels):
        raise ValueError("metadata length does not match labels length")
    if groups is not None and groups not in metadata.columns:
        raise ValueError(f"groups column '{groups}' not in metadata")

    preds = (probs >= 0.5).astype(int) if probs.ndim == 1 else probs.argmax(axis=1)
    fairness = fairness_audit(
        labels == positive_class, preds == positive_class, metadata[slice_by],
        classes=[False, True], positive_class=True,
        groups=None if groups is None else metadata[groups],
        n_boot=DEFAULT_BOOTSTRAP if n_boot is None else n_boot,
        min_group_size=MIN_GROUP_SIZE if min_group_size is None else min_group_size, seed=seed)

    blocks: dict[str, dict] = {}
    for value, idx in metadata.groupby(slice_by, dropna=True, sort=True).indices.items():
        name = str(value)
        blocks[name] = dict(_metrics_block(probs[idx], labels[idx], positive_class),
                            eligible=fairness["groups"][name]["eligible"])

    return {
        "slice_by": slice_by,
        "global": _metrics_block(probs, labels, positive_class),
        "groups": blocks,
        "gaps": {key: fairness["gaps"][name]["value"] for key, name in _LEGACY_GAP_NAMES},
        "fairness": fairness,
    }


_LEGACY_GAP_NAMES = [("eo_tpr", "equal_opportunity"), ("eo_fpr", "predictive_equality"),
                     ("demographic_parity", "demographic_parity"), ("equalized_odds", "equalized_odds")]


# ---------------------------------------------------------------------------
# Fairness gaps with uncertainty
# ---------------------------------------------------------------------------

DEFAULT_BOOTSTRAP = 1000      # resamples behind every confidence interval
MAX_BOOTSTRAP = 10000
MIN_GROUP_SIZE = 20           # groups with fewer subjects (rows without Group By) are shown but not compared
DISPARITY_THRESHOLD = 0.15    # a rate gap above this is flagged, provided its interval excludes 0
BOOTSTRAP_BUDGET_BYTES = 64 * 2**20  # resampling weights generated at once

_METRIC_LABELS = {"tpr": "TPR", "fpr": "FPR", "selection_rate": "selection rate", "accuracy": "accuracy",
                  "balanced_accuracy": "balanced accuracy", "rmse": "RMSE", "mae": "MAE", "mean_error": "mean error"}


def metric_label(metric: str) -> str:
    """A readable name for a metric key, e.g. "recall:AD" -> "recall of AD"."""
    if ":" in metric:
        kind, cls = metric.split(":", 1)
        return f"{_METRIC_LABELS.get(kind, kind)} of {cls}"
    return _METRIC_LABELS.get(metric, metric)


_RATE_GAPS = {
    "binary": [("equal_opportunity", "tpr"), ("predictive_equality", "fpr"),
               ("demographic_parity", "selection_rate"), ("accuracy", "accuracy")],
    "regression": [("rmse", "rmse"), ("mae", "mae"), ("mean_error", "mean_error")],
}


def fairness_audit(y_true, y_pred, slices=None, *, task: str = "classification", classes=None,
                   positive_class=None, groups=None, n_boot: int = DEFAULT_BOOTSTRAP,
                   min_group_size: int = MIN_GROUP_SIZE, confidence: float = 0.95,
                   threshold: float = DISPARITY_THRESHOLD, seed: int = 42, progress=None) -> dict:
    """
    Performance per group of `slices` (e.g. each row's sex) and the gaps between groups, each
    with a percentile bootstrap confidence interval.

    Classification takes predicted labels; with two `classes` it is binary (`positive_class`
    defaults to the last class), with more it is multiclass. Regression (task="regression")
    takes predicted values. Rows without a slice value count only towards the global figures.

    `groups` (subject IDs, i.e. the Group By column) makes the bootstrap resample whole
    subjects, so repeat scans of one person are not counted as independent evidence; without
    it rows are resampled. Every metric is computed from counts per resampled unit, so a
    resample costs one sparse product, not a pass over the rows.

    A gap is the difference between the best and the worst eligible group, the pair being fixed
    by the point estimate; its interval is that pair's difference across the same resamples.
    Groups with fewer than `min_group_size` units are reported but not eligible. A rate gap is
    flagged when it exceeds `threshold` and its interval excludes 0, and marked as possible
    noise when it exceeds `threshold` but the interval includes 0. Regression gaps are in the
    target's units and are flagged when their interval excludes 0.

    `progress`, if given, is called with the fraction of resamples done (0-1); raising from it
    (e.g. core.cancel.Cancelled) stops the audit.

    Returns {task, classes, positive_class, bootstrap, min_group_size, global, groups, gaps,
    notes}. Each metric is {"value", "ci"}; "ci" is None without resamples.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    if task not in ("classification", "regression"):
        raise ValueError(f"task must be 'classification' or 'regression', not {task!r}")
    if task == "regression" and y_pred.size == n:
        y_pred = y_pred.reshape(n)
    if len(y_pred) != n or y_pred.ndim != 1:
        raise ValueError(f"expected one prediction per row: {n} rows, predictions of shape {y_pred.shape}")
    if not 0 <= int(n_boot) <= MAX_BOOTSTRAP:
        raise ValueError(f"n_boot must be between 0 and {MAX_BOOTSTRAP}")
    if int(min_group_size) < 1:
        raise ValueError("min_group_size must be at least 1")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    n_boot, min_group_size = int(n_boot), int(min_group_size)
    notes: list[str] = []

    if task == "regression":
        kind, positive, classes = "regression", None, None
        truth, pred = y_true.astype(float), y_pred.astype(float)
        keep = np.isfinite(truth) & np.isfinite(pred)
        dropped = "a missing target or prediction"
    else:
        classes = _class_list(y_true, y_pred, classes)
        if len(classes) < 2:
            raise ValueError("classification needs at least two classes")
        index = pd.Index(classes)
        t, p = index.get_indexer(pd.Index(y_true)), index.get_indexer(pd.Index(y_pred))
        keep = (t >= 0) & (p >= 0)
        dropped = "a missing label or a label outside the classes"
        kind = "binary" if len(classes) == 2 else "multiclass"
        positive = None
        if kind == "binary":
            positive = classes[-1] if positive_class is None else positive_class
            if positive not in classes:
                raise ValueError(f"positive class {positive!r} is not one of {classes}")
    if not keep.all():
        notes.append(f"{int((~keep).sum())} rows with {dropped} are left out.")
    if not keep.any():
        raise ValueError("no rows left to analyse")

    slice_codes, slice_names = _slice_codes(slices, n, keep, notes)
    unit_codes, n_units = _unit_codes(groups, n, keep, notes)
    unit = "group" if groups is not None else "row"
    unit_word = "subjects" if unit == "group" else "rows"
    n_slices = len(slice_names)
    cells = n_slices + 1  # the last cell holds rows without a slice value

    if kind == "regression":
        truth, pred = truth[keep], pred[keep]
        error, centred = pred - truth, truth - truth.mean()  # R^2 is shift-invariant; centring keeps the sums exact
        values = np.column_stack([np.ones_like(error), error, error ** 2, np.abs(error), centred, centred ** 2])
        width = values.shape[1]
        columns = (slice_codes * width)[:, None] + np.arange(width)
        counts = sparse.csr_matrix((values.ravel(), (np.repeat(unit_codes, width), columns.ravel())),
                                   shape=(n_units, cells * width))
        metrics_of = _regression_metrics
    else:
        k = len(classes)
        width = k * k
        columns = slice_codes * width + t[keep] * k + p[keep]
        counts = sparse.csr_matrix((np.ones(len(columns)), (unit_codes, columns)), shape=(n_units, cells * width))
        positive_index = None if positive is None else classes.index(positive)
        metrics_of = lambda totals: _classification_metrics(totals, classes, positive_index)  # noqa: E731

    def totals_for(weights):
        """(resamples, cells + 1, width): per-slice totals, then the global total."""
        per_slice = np.asarray(counts.T @ weights.T).T.reshape(len(weights), cells, width)
        return np.concatenate([per_slice, per_slice.sum(axis=1, keepdims=True)], axis=1)

    point_totals = totals_for(np.ones((1, n_units)))
    point = {name: values[0] for name, values in metrics_of(point_totals).items()}
    replicates = _bootstrap(totals_for, metrics_of, n_units, n_boot, seed, progress)
    tails = [100 * (1 - confidence) / 2, 100 * (1 + confidence) / 2]

    def summarise(cell):
        return {name: {"value": _finite_or_nan(point[name][cell]),
                       "ci": _interval(replicates, name, cell, point[name][cell], tails)}
                for name in point}

    rows_per_slice = np.bincount(slice_codes, minlength=cells)
    units_per_slice = np.bincount(np.unique(unit_codes.astype(np.int64) * cells + slice_codes) % cells,
                                  minlength=cells)
    result_groups = {}
    for s, name in enumerate(slice_names):
        result_groups[name] = {"n": int(rows_per_slice[s]), "units": int(units_per_slice[s]),
                               "eligible": bool(units_per_slice[s] >= min_group_size),
                               "metrics": summarise(s)}
    result_global = {"n": int(keep.sum()), "units": int(n_units), "metrics": summarise(cells)}
    if kind != "regression":
        k = len(classes)
        for s, block in enumerate(list(result_groups.values()) + [result_global]):
            cell = s if s < n_slices else cells
            block["confusion"] = point_totals[0, cell].reshape(k, k).astype(int).tolist()

    gap_specs = _RATE_GAPS.get(kind) or (
        [("accuracy", "accuracy"), ("balanced_accuracy", "balanced_accuracy")]
        + [(f"recall:{c}", f"recall:{c}") for c in classes]
        + [(f"selection_rate:{c}", f"selection_rate:{c}") for c in classes])
    eligible = [s for s in range(n_slices) if units_per_slice[s] >= min_group_size]
    gaps = {}
    if n_slices:
        for gap_name, metric in gap_specs:
            gaps[gap_name] = _gap(metric, point[metric], replicates.get(metric), eligible, slice_names,
                                  tails, kind, threshold, min_group_size, unit_word, notes)
        if kind == "binary":
            parts = [gaps["equal_opportunity"], gaps["predictive_equality"]]
            measured = [g for g in parts if np.isfinite(g["value"])]
            gaps["equalized_odds"] = (dict(max(measured, key=lambda g: g["value"])) if measured else
                                      dict(parts[0], reason="neither the TPR nor the FPR gap could be measured"))
        if len(eligible) > 2:
            notes.append(f"With {len(eligible)} groups, each gap is the largest of several differences, so it "
                         "tends to overstate the typical one; treat it as a screen.")
    if n_boot == 0:
        notes.append("No resamples were drawn, so there are no intervals"
                     + (" and rate gaps are flagged on size alone." if kind != "regression" else "."))

    return {
        "task": kind,
        "classes": classes,
        "positive_class": positive,
        "bootstrap": {"replicates": n_boot, "unit": unit, "units": int(n_units),
                      "confidence": confidence, "seed": seed},
        "min_group_size": min_group_size,
        "global": result_global,
        "groups": result_groups,
        "gaps": gaps,
        "notes": notes,
    }


def _class_list(y_true, y_pred, classes):
    if classes is not None:
        return list(classes)
    present = pd.unique(np.concatenate([pd.Series(y_true).dropna().to_numpy(),
                                        pd.Series(y_pred).dropna().to_numpy()]))
    try:
        return sorted(present)
    except TypeError:
        return sorted(present, key=str)


def _slice_codes(slices, n, keep, notes):
    """Codes 0..S-1 for the kept rows' groups (sorted), S for rows without one, and the names."""
    if slices is None:
        return np.zeros(int(keep.sum()), dtype=np.int64), []
    values = pd.Series(np.asarray(slices, dtype=object) if not isinstance(slices, pd.Series) else slices.to_numpy())
    if len(values) != n:
        raise ValueError(f"slices has {len(values)} values for {n} rows")
    values = values[keep].reset_index(drop=True)
    try:
        codes, uniques = pd.factorize(values, sort=True)
    except TypeError:  # mixed types that cannot be ordered
        codes, uniques = pd.factorize(values.where(values.isna(), values.astype(str)), sort=True)
    names = [str(u) for u in uniques]
    if len(set(names)) != len(names):
        raise ValueError("two slice values print the same; convert the column to one type first")
    missing = codes < 0
    if missing.any():
        notes.append(f"{int(missing.sum())} rows have no value to slice by; they count only towards the global figures.")
    return np.where(missing, len(names), codes).astype(np.int64), names


def _unit_codes(groups, n, keep, notes):
    """The bootstrap unit of each kept row: its subject when `groups` is given, else itself."""
    if groups is None:
        m = int(keep.sum())
        return np.arange(m, dtype=np.int64), m
    values = pd.Series(np.asarray(groups, dtype=object) if not isinstance(groups, pd.Series) else groups.to_numpy())
    if len(values) != n:
        raise ValueError(f"groups has {len(values)} values for {n} rows")
    codes, uniques = pd.factorize(values[keep].reset_index(drop=True))
    missing = codes < 0
    if missing.any():
        notes.append(f"{int(missing.sum())} rows have no Group By value; each is resampled on its own.")
        codes = codes.copy()
        codes[missing] = len(uniques) + np.arange(int(missing.sum()))
    return codes.astype(np.int64), len(uniques) + int(missing.sum())


def _bootstrap(totals_for, metrics_of, n_units, n_boot, seed, progress=None):
    """Metric values over `n_boot` resamples of the units: {name: (n_boot, cells + 1)}."""
    if n_boot == 0:
        return {}
    rng = np.random.default_rng(seed)
    block = max(1, min(n_boot, BOOTSTRAP_BUDGET_BYTES // (24 * n_units)))
    out = {}
    for start in range(0, n_boot, block):
        b = min(block, n_boot - start)
        draws = rng.integers(0, n_units, size=(b, n_units)) + (np.arange(b) * n_units)[:, None]
        weights = np.bincount(draws.ravel(), minlength=b * n_units).reshape(b, n_units).astype(float)
        for name, values in metrics_of(totals_for(weights)).items():
            out.setdefault(name, np.empty((n_boot,) + values.shape[1:]))[start:start + b] = values
        if progress is not None:
            progress((start + b) / n_boot)
    return out


def _ratio(num, den):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def _classification_metrics(totals, classes, positive):
    """Rates from confusion counts (rows: truth, columns: prediction) of shape (..., k * k)."""
    k = len(classes)
    c = totals.reshape(totals.shape[:-1] + (k, k))
    n = c.sum(axis=(-2, -1))
    diag = np.diagonal(c, axis1=-2, axis2=-1)
    truth, predicted = c.sum(axis=-1), c.sum(axis=-2)
    out = {"accuracy": _ratio(diag.sum(axis=-1), n)}
    if positive is not None:
        tp, cases, flagged = diag[..., positive], truth[..., positive], predicted[..., positive]
        fp, fn = flagged - tp, cases - tp
        negatives = n - cases
        out.update(tpr=_ratio(tp, cases), fnr=_ratio(fn, cases), fpr=_ratio(fp, negatives),
                   tnr=_ratio(negatives - fp, negatives), precision=_ratio(tp, flagged),
                   f1=_ratio(2 * tp, 2 * tp + fp + fn), selection_rate=_ratio(flagged, n))
    else:
        recall = _ratio(diag, truth)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # a resample can leave no class with cases
            out["balanced_accuracy"] = np.nanmean(recall, axis=-1)
        for i, name in enumerate(classes):
            out[f"recall:{name}"] = recall[..., i]
            out[f"selection_rate:{name}"] = _ratio(predicted[..., i], n)
    return out


def _regression_metrics(totals):
    """Error summaries from per-cell sums of (1, e, e^2, |e|, y, y^2), e = prediction - truth."""
    n, e, ee, abs_e, y, yy = np.moveaxis(totals, -1, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        spread = yy - _ratio(y * y, n)
        r2 = np.where(spread > 1e-10 * np.maximum(yy, 1e-300), 1 - ee / np.where(spread > 0, spread, 1), np.nan)
    return {"rmse": np.sqrt(_ratio(ee, n)), "mae": _ratio(abs_e, n), "mean_error": _ratio(e, n), "r2": r2}


def _finite_or_nan(value):
    value = float(value)
    return value if np.isfinite(value) else float("nan")


def _interval(replicates, name, cell, point_value, tails):
    if name not in replicates or not np.isfinite(point_value):
        return None
    values = replicates[name][:, cell]
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    lo, hi = np.percentile(values, tails)
    return float(lo), float(hi)


def _gap(metric, point, replicates, eligible, names, tails, kind, threshold, min_group_size, unit_word, notes):
    measured = [s for s in eligible if np.isfinite(point[s])]
    if len(measured) < 2:
        return {"metric": metric, "value": float("nan"), "ci": None, "high": None, "low": None,
                "flagged": False, "noise": False,
                "reason": f"fewer than two groups with at least {min_group_size} {unit_word} and a defined "
                          f"{metric_label(metric)}"}
    order = sorted(measured, key=lambda s: point[s])
    low, high = order[0], order[-1]
    value = float(point[high] - point[low])
    ci = None
    if replicates is not None:
        diffs = replicates[:, high] - replicates[:, low]
        defined = np.isfinite(diffs)
        if defined.any():
            ci = tuple(float(v) for v in np.percentile(diffs[defined], tails))
        if defined.mean() < 0.99:
            label = metric_label(metric)
            notes.append(f"{label[:1].upper()}{label[1:]}: {1 - defined.mean():.0%} of resamples left "
                         f"{names[high]} or {names[low]} without cases to measure it on; its interval uses the rest.")
    excludes_zero = ci is not None and (ci[0] > 0 or ci[1] < 0)
    if kind == "regression":
        flagged, noise = excludes_zero, False
    else:
        large = value > threshold
        flagged = large and (excludes_zero or replicates is None)
        noise = large and not flagged
    return {"metric": metric, "value": value, "ci": ci, "high": names[high], "low": names[low],
            "flagged": bool(flagged), "noise": bool(noise)}


# ---------------------------------------------------------------------------
# Stage 3: linear probing
# ---------------------------------------------------------------------------

DEFAULT_CONTROLS = 50       # shuffled-label repeats when a control is requested
MAX_CONTROLS = 1000
REGRESSION_MIN_UNIQUE = 10  # numeric targets with more distinct values get a regression probe
PROBE_MODELS = ("lda", "logistic", "ridge")  # the first is the default
STREAMED_MODELS = ("lda", "ridge")
RIDGE_ALPHA = 1.0
LDA_SHRINKAGE = 0.1           # pull towards the average feature variance; keeps LDA stable when features ~ rows
LDA_DIAGONAL_FLOOR = 1e-9     # in standardised units: keeps the covariance invertible in degenerate cases
STREAM_MAX_BYTES = 4 * 2**30  # running totals larger than this are refused (D x D per fold, ~10,000 features)
CHUNK_BYTES = 64 * 2**20      # rows are streamed in blocks of about this size
SCORE_BUDGET_BYTES = 256 * 2**20  # held-out scores are kept for at most this many bytes' worth of shuffles at once
WOODBURY_MIN_SETS = 6         # from this many label sets (5+ shuffles), LDA factorises each fold once and
                              # updates per set. Measured at 100k rows: at 2,048 features this is faster from
                              # 5 shuffles (21.5 s vs 23.6 s) and 2.2x faster at 20; at 512 both take 2-3 s.


def linear_probe(features, target, groups=None, n_splits: int = 5, seed: int = 42,
                 n_controls: int = 0, task: str = "auto", model: str = "lda",
                 chunk_rows: int | None = None, progress=None) -> dict:
    """Cross-validated linear probe: can `target` be read linearly out of `features`?

    `features` is an array, or anything with `.shape` that can be sliced by rows, such as
    an open h5py dataset (see AuditCache.open_layer), so it can stay on disk.

    * Classification reports ROC-AUC (one-vs-rest macro average for >2 classes) and
      balanced accuracy, whose chance level is 1/k. `model` picks the classifier:
        - "lda" (default): shrinkage linear discriminant analysis. Streamed and exact,
          and handles several classes, including ordered ones, without masking.
        - "logistic": class-balanced (multinomial) logistic regression, the classic
          probe. Loads the layer into memory, and every control is a full refit.
        - "ridge": ridge classifier (least squares on -1/+1 targets). Streamed and exact;
          fine for two classes, but with three or more a class lying between others can
          be masked.
      The streamed models build their exact solution from running totals: the layer is
      read in row chunks (once for the totals, then to score held-out rows) and never
      loaded whole. Their memory grows with features^2, not rows: up to about 10,000
      features per row.
    * Regression (ridge, R^2) is used for numeric targets with more than 10 distinct
      values; it always streams.
    * `groups` (e.g. subject IDs) keeps all of a group's rows in one fold. Without it, a
      subject with several scans can sit in both training and test folds and the probe
      can score by recognising them, so pass it whenever subjects repeat.
    * Standardisation is fitted inside each training fold.
    * `n_controls` (0-1000) re-runs the probe that many times on shuffled labels (on the
      same folds) and reports how often chance matched the real score; the smallest p it
      can show is 1/(n_controls + 1). With `groups`, whole groups' labels are shuffled
      between groups when the label is constant within each group. (scikit-learn's
      permutation_test_score shuffles *within* groups instead.)

    `progress`, if given, is called with the fraction of the work done (0-1) after each block
    of rows read and each model fitted; raising from it (e.g. core.cancel.Cancelled) stops
    the probe.

    Rows with a missing target are dropped. Data that cannot be probed is reported in
    `error` instead of raising.
    """
    if model not in PROBE_MODELS:
        raise ValueError(f"Unknown probe model {model!r}; expected one of {PROBE_MODELS}")
    if isinstance(n_controls, bool) or not isinstance(n_controls, (int, np.integer)) or not 0 <= n_controls <= MAX_CONTROLS:
        raise ValueError(f"n_controls must be a whole number from 0 to {MAX_CONTROLS}, got {n_controls!r}")
    n_controls = int(n_controls)
    if not hasattr(features, "shape"):
        features = np.asarray(features, dtype=float)
    y = (target if isinstance(target, pd.Series) else pd.Series(np.asarray(target))).reset_index(drop=True)
    g = None if groups is None else pd.Series(groups).reset_index(drop=True)
    if features.shape[0] != len(y) or (g is not None and len(g) != len(y)):
        raise ValueError("features, target and groups must have one entry per row")

    notes: list[str] = []
    keep = ~y.isna().to_numpy()
    dropped = int((~keep).sum())
    if dropped:
        notes.append(f"{dropped} rows with a missing target were left out.")
    source = _RowSource(features, keep, chunk_rows, _Ticker(progress))
    y = y[keep].reset_index(drop=True)
    if g is not None:
        g = g[keep].reset_index(drop=True)
        if g.isna().any():
            notes.append(f"{int(g.isna().sum())} rows with no group were treated as groups of their own.")
            g = g.astype(object).where(g.notna(), [f"__row_{i}" for i in range(len(g))])
        group_codes = pd.factorize(g)[0]
    else:
        group_codes = None

    if task == "auto":
        numeric = pd.api.types.is_numeric_dtype(y)
        task = "regression" if numeric and y.nunique() > REGRESSION_MIN_UNIQUE else "classification"

    result = {
        "task": task, "metric": "R²" if task == "regression" else "AUC",
        "model": "ridge" if task == "regression" else model,
        "score_mean": float("nan"), "score_std": 0.0, "folds": [],
        "n_samples": int(len(y)), "dropped_missing": dropped, "n_splits": None,
        "grouped": g is not None, "n_groups": None if g is None else int(g.nunique()),
        "control": None, "notes": notes, "error": None,
    }
    if task == "regression":
        _probe_regression(source, y.astype(float).to_numpy(), group_codes, n_splits, seed, n_controls, result)
    else:
        _probe_classification(source, y.astype(str).to_numpy(), group_codes, n_splits, seed, n_controls, result)
    source.ticker.finish()
    return result


class _Ticker:
    """Counts finished units of work and reports the fraction done to an optional callback."""

    def __init__(self, progress=None):
        self.progress, self.done, self.total = progress, 0, 1

    def plan(self, total):
        self.done, self.total = 0, max(int(total), 1)

    def tick(self, units=1):
        self.done += units
        if self.progress is not None:
            self.progress(min(self.done / self.total, 1.0))

    def finish(self):
        if self.progress is not None:
            self.progress(1.0)


class _RowSource:
    """The kept rows of `features` as float64 blocks, read lazily and in order. Each block read
    is one tick of `ticker`."""

    def __init__(self, features, keep, chunk_rows=None, ticker=None):
        self.features, self.keep = features, keep
        self.width = int(np.prod(features.shape[1:]))
        self.chunk_rows = chunk_rows or max(1, CHUNK_BYTES // (8 * max(self.width, 1)))
        self.ticker = ticker or _Ticker()

    @property
    def n_chunks(self):
        return -(-self.features.shape[0] // self.chunk_rows)

    def chunks(self):
        """Yield (positions among the kept rows, rows x width float64 block)."""
        n_total = self.features.shape[0]
        kept_position = np.cumsum(self.keep) - 1
        for start in range(0, n_total, self.chunk_rows):
            stop = min(start + self.chunk_rows, n_total)
            mask = self.keep[start:stop]
            if mask.any():
                block = np.asarray(self.features[start:stop], dtype=np.float64).reshape(stop - start, self.width)
                yield kept_position[start:stop][mask], block[mask]
            self.ticker.tick()

    def load(self, dtype=np.float32):
        """All kept rows in one array, filled chunk by chunk (no second copy). float32 by default:
        the logistic probe's solver converts each training fold to float64 itself."""
        X = np.empty((int(self.keep.sum()), self.width), dtype=dtype)
        for positions, block in self.chunks():
            X[positions] = block
        return X


def _probe_classification(source, labels, groups, n_splits, seed, n_controls, result):
    classes, y = np.unique(labels, return_inverse=True)
    n_classes = len(classes)
    model = result["model"]
    result.update(classes=[str(c) for c in classes], chance_balanced_accuracy=1.0 / n_classes,
                  auc_mean=float("nan"), auc_std=0.0,
                  balanced_accuracy_mean=float("nan"), balanced_accuracy_std=0.0)
    if n_classes < 2:
        result["error"] = "The target has fewer than 2 classes, so there is nothing to probe."
        return

    # Each class must reach every fold: count its rows, or its groups when grouping.
    if groups is None:
        per_class, unit = np.bincount(y), "samples"
    else:
        per_class, unit = np.array([len(np.unique(groups[y == k])) for k in range(n_classes)]), "groups"
    n_splits = min(n_splits, int(per_class.min()))
    if n_splits < 2:
        rare = classes[int(np.argmin(per_class))]
        result["error"] = (f"Class '{rare}' has too few {unit} ({int(per_class.min())}) to cross-validate; "
                           "it needs at least 2.")
        return
    if model in STREAMED_MODELS and _stream_memory_error(source.width, n_splits, model, result):
        return
    result["n_splits"] = n_splits

    if groups is None:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    else:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(cv.split(np.zeros(len(y)), y, groups))
    code_type = np.uint8 if n_classes <= 255 else np.int32
    label_sets, shuffle_unit = _label_sets(y, groups, n_controls, seed, result["notes"], code_type)

    ticker = source.ticker
    if model == "logistic":
        ticker.plan(source.n_chunks + len(label_sets) * len(splits))
        X = source.load()
        aucs, bal_accs = _logistic_folds(X, y, splits, n_classes, ticker)
        control_aucs = [_logistic_folds(X, label_sets[p].astype(int), splits, n_classes, ticker)[0]
                        for p in range(1, len(label_sets))]
    else:
        n_cols = 1 if n_classes == 2 else n_classes
        ticker.plan(source.n_chunks * (1 + _score_passes(len(y), n_cols, len(label_sets))) + len(splits))
        fold_aucs, bal_accs = _streamed_classification(source, splits, label_sets, n_classes, model)
        aucs, control_aucs = fold_aucs[0], fold_aucs[1:]

    if np.all(np.isnan(aucs)):
        result["error"] = "No fold contained more than one class, so AUC could not be computed."
        return
    if np.isnan(aucs).any():
        result["notes"].append(f"{int(np.isnan(aucs).sum())} folds held a single class and were left out of the AUC.")
    result.update(auc_mean=float(np.nanmean(aucs)), auc_std=float(np.nanstd(aucs)), folds=[float(a) for a in aucs],
                  balanced_accuracy_mean=float(np.mean(bal_accs)), balanced_accuracy_std=float(np.std(bal_accs)))
    result.update(score_mean=result["auc_mean"], score_std=result["auc_std"])
    if n_controls:
        means = [float(np.nanmean(a)) if not np.all(np.isnan(a)) else float("nan") for a in control_aucs]
        result["control"] = _control_summary(means, result["auc_mean"], shuffle_unit)


def _balanced_accuracy(y_true, predicted):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # tiny folds: "y_pred contains classes not in y_true"
        return float(balanced_accuracy_score(y_true, predicted))


def _logistic_folds(X, y, splits, n_classes, ticker=None):
    aucs, bal_accs = [], []
    for train, test in splits:
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            model.fit(X[train], y[train])
        bal_accs.append(_balanced_accuracy(y[test], model.predict(X[test])))
        proba = np.zeros((len(test), n_classes))
        proba[:, model[-1].classes_] = model.predict_proba(X[test])
        aucs.append(_macro_auc(y[test], proba, n_classes))
        if ticker is not None:
            ticker.tick()
    return aucs, bal_accs


def _macro_auc(y_true, proba, n_classes):
    """ROC-AUC, one-vs-rest averaged over the classes present in this fold; NaN if only one is."""
    present = np.unique(y_true)
    if len(present) < 2 or np.isnan(proba).any():
        return float("nan")
    if n_classes == 2:
        return _auc(y_true == 1, proba[:, 1])
    return float(np.mean([_auc(y_true == k, proba[:, k]) for k in present]))


def _auc(is_positive, scores):
    """Binary ROC-AUC from ranks (the Mann-Whitney statistic, tied scores averaged): the same
    number as sklearn's roc_auc_score without its per-call overhead, which adds up over
    hundreds of controls x folds."""
    n_pos = int(is_positive.sum())
    n_neg = len(is_positive) - n_pos
    return float((rankdata(scores)[is_positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# --- streamed models: running totals (pass 1) -> exact per-fold solve -> held-out scores (pass 2)

def _stream_memory_error(width, n_splits, model, result):
    need = n_splits * width * width * 8
    if need > STREAM_MAX_BYTES:
        result["error"] = (f"The streamed {model} probe would need {need / 2**30:.1f} GB of running totals for "
                           f"{width} features (it handles up to about 10,000). Use a narrower layer, "
                           "e.g. a pooled one, or the logistic probe.")
        return True
    return False


def _streamed_classification(source, splits, label_sets, n_classes, model):
    """Fold AUCs for every label set (row 0 real, the rest controls) and the real set's balanced accuracy."""
    n_sets, n_cols = len(label_sets), (1 if n_classes == 2 else n_classes)
    if model == "lda":
        totals = _accumulate(source, splits, class_labels=label_sets, n_classes=n_classes)
        models, thresholds = _solve_lda(totals, n_classes, source.ticker)
    else:
        totals = _accumulate(source, splits, class_labels=label_sets[:1], n_classes=n_classes,
                             targets=lambda pos: _class_targets(label_sets[:, pos], n_classes),
                             n_targets=n_sets * n_cols)
        models = _solve_ridge(totals, source.ticker)
        thresholds = _ridge_thresholds(totals, models, n_cols)

    fold_aucs = np.full((n_sets, len(splits)), np.nan)
    bal_accs = None
    for first, scores in _held_out_scores(source, totals, models, n_cols, n_sets):
        count = scores.shape[1] // n_cols
        fold_aucs[first:first + count] = _fold_aucs(label_sets[first:first + count], scores, splits, n_classes, model)
        if first == 0:
            real_scores, labels = scores[:, :n_cols], label_sets[0].astype(int)
            bal_accs = [_thresholded_balanced_accuracy(labels[test], real_scores[test], thresholds[f])
                        for f, (_, test) in enumerate(splits)]
    return fold_aucs, bal_accs


def _accumulate(source, splits, class_labels=None, n_classes=0, targets=None, n_targets=0):
    """Pass 1: per-fold running totals of the features, of their products with each target
    column (`targets(positions)` builds them per chunk), and of their sums within each class
    of each label set in `class_labels` (sets x rows)."""
    n_rows, n_folds = sum(len(test) for _, test in splits), len(splits)
    fold_of = np.empty(n_rows, dtype=int)
    for f, (_, test) in enumerate(splits):
        fold_of[test] = f
    width = source.width
    n_label_sets = 0 if class_labels is None else len(class_labels)
    t = SimpleNamespace(
        fold_of=fold_of, n_rows=n_rows, shift=np.zeros(width),
        count=np.zeros(n_folds), sum_x=np.zeros((n_folds, width)), sum_xx=np.zeros((n_folds, width, width)),
        sum_xy=np.zeros((n_folds, width, n_targets)), sum_y=np.zeros((n_folds, n_targets)),
        class_sum=np.zeros((n_label_sets, n_folds, n_classes, width)),
        class_count=np.zeros((n_label_sets, n_folds, n_classes)),
    )
    first = True
    for positions, X in source.chunks():
        if first:
            t.shift, first = X.mean(axis=0), False  # sums relative to a typical row stay numerically stable
        X = X - t.shift
        here = fold_of[positions]
        for f in np.unique(here):
            rows = here == f
            Xf, pos = X[rows], positions[rows]
            t.count[f] += len(Xf)
            t.sum_x[f] += Xf.sum(axis=0)
            t.sum_xx[f] += Xf.T @ Xf
            if n_targets:
                T = targets(pos)
                t.sum_xy[f] += Xf.T @ T
                t.sum_y[f] += T.sum(axis=0)
            for start in range(0, n_label_sets, 128):  # a slice of label sets at a time bounds the temporaries
                codes = class_labels[start:start + 128, pos]
                for k in range(n_classes):
                    member = (codes == k).astype(float)
                    t.class_sum[start:start + 128, f, k] += member @ Xf
                    t.class_count[start:start + 128, f, k] += member.sum(axis=1)
    return t


def _train_stats(t, f):
    """Fold f's training rows: (mask of training folds, n, mean, centred scatter, StandardScaler scale)."""
    train = np.arange(len(t.count)) != f
    n = t.count[train].sum()
    mean = t.sum_x[train].sum(axis=0) / n
    scatter = t.sum_xx[train].sum(axis=0) - n * np.outer(mean, mean)
    scale = np.sqrt(np.clip(np.diag(scatter) / n, 0, None))
    scale[scale < 10 * np.finfo(float).eps] = 1.0  # as StandardScaler treats constant features
    return train, n, mean, scatter, scale


def _solve_ridge(t, ticker=None):
    """Per fold: ridge on standardised features with an intercept, for every target column."""
    models = []
    width = t.sum_x.shape[1]
    for f in range(len(t.count)):
        train, n, mean, scatter, scale = _train_stats(t, f)
        y_mean = t.sum_y[train].sum(axis=0) / n
        cross = t.sum_xy[train].sum(axis=0) - n * np.outer(mean, y_mean)
        system = scatter / np.outer(scale, scale)
        system[np.diag_indices(width)] += RIDGE_ALPHA
        weights = np.linalg.solve(system, cross / scale[:, None]) / scale[:, None]  # acts on shifted x
        models.append((mean, weights, y_mean))
        if ticker is not None:
            ticker.tick()
    return models


def _solve_lda(t, n_classes, ticker=None):
    """Per fold and label set: shrinkage LDA on standardised features, as scikit-learn's
    LinearDiscriminantAnalysis(solver="lsqr", shrinkage=LDA_SHRINKAGE) computes it.

    Each label set's covariance is the label-independent total scatter minus a rank-k
    between-class term, plus a scalar on the diagonal. So with many sets (controls), each
    fold's total scatter is eigendecomposed once and every set is a cheap Woodbury update
    instead of its own features x features solve; both give the same coefficients.
    Returns per fold (mean, weights, offsets) giving each set's scores (binary: one column,
    the class-1 minus class-0 discriminant) and the real set's equal-prior thresholds.
    """
    n_sets, width, binary = t.class_sum.shape[0], t.sum_x.shape[1], n_classes == 2
    n_cols = 1 if binary else n_classes
    woodbury = n_sets >= WOODBURY_MIN_SETS
    keep = 1 - LDA_SHRINKAGE
    models, thresholds = [], []
    for f in range(len(t.count)):
        train, n, mean, scatter, scale = _train_stats(t, f)
        total_z = scatter / np.outer(scale, scale)
        trace_total = np.trace(total_z)
        if woodbury:
            eigenvalues, V = np.linalg.eigh(total_z)
            eigenvalues = np.clip(eigenvalues, 0, None)
        weights, offsets = np.zeros((width, n_sets * n_cols)), np.zeros(n_sets * n_cols)
        threshold = None
        for p in range(n_sets):
            cols = slice(p * n_cols, (p + 1) * n_cols)
            class_n = t.class_count[p, train].sum(axis=0)
            if (class_n == 0).any():  # a class missing from this training fold (possible for a control)
                offsets[cols] = np.nan
                continue
            means_z = (t.class_sum[p, train].sum(axis=0) / class_n[:, None] - mean) / scale  # classes x features
            diagonal = LDA_SHRINKAGE * (trace_total - (class_n * (means_z ** 2).sum(axis=1)).sum()) / (n * width)
            # If every feature is constant within each class, the covariance and its shrinkage target are
            # both zero and scikit-learn's solution collapses to 0 (AUC 0.5 for a perfect separator).
            # A tiny floor keeps it invertible; on real data it is ~1e-8 of the shrinkage term.
            diagonal = max(diagonal, LDA_DIAGONAL_FLOOR)
            if woodbury and diagonal > 1e-6:
                # (keep/n * total - U U^T + diagonal I)^-1 means^T, in the eigenbasis of total
                R = V.T @ means_z.T
                a = keep / n * eigenvalues + diagonal
                U = R * np.sqrt(keep * class_n / n)
                a_inv_R, a_inv_U = R / a[:, None], U / a[:, None]
                inner = np.eye(n_classes) - U.T @ a_inv_U
                coef = (V @ (a_inv_R + a_inv_U @ np.linalg.solve(inner, U.T @ a_inv_R))).T
            else:
                within = (total_z - (means_z.T * class_n) @ means_z) / n
                shrunk = keep * within + diagonal * np.eye(width)
                coef = _solve_or_least_squares(shrunk, means_z.T).T
            log_prior = np.log(class_n / n)
            intercept = -0.5 * np.sum(means_z * coef, axis=1) + log_prior
            w = (coef / scale).T
            if binary:
                weights[:, cols] = (w[:, 1] - w[:, 0])[:, None]
                offsets[cols] = intercept[1] - intercept[0]
                set_threshold = log_prior[1] - log_prior[0]
            else:
                weights[:, cols], offsets[cols], set_threshold = w, intercept, log_prior
            if p == 0:
                threshold = set_threshold  # predicting with equal priors does not favour the majority
        models.append((mean, weights, offsets))
        thresholds.append(threshold)
        if ticker is not None:
            ticker.tick()
    return models, thresholds


def _solve_or_least_squares(A, B):
    """Solve A X = B. A singular A (e.g. classes split perfectly by a feature that is constant
    within each class) falls back to the least-squares solution scikit-learn's lsqr solver gives."""
    try:
        return np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, B, rcond=None)[0]


def _class_targets(codes, n_classes):
    """-1/+1 targets for every label set (sets x rows codes -> rows x sets*columns), as
    scikit-learn's RidgeClassifier builds them."""
    if n_classes == 2:
        return np.where(codes.T == 1, 1.0, -1.0)
    rows = codes.shape[1]
    return np.where(codes.T[:, :, None] == np.arange(n_classes), 1.0, -1.0).reshape(rows, -1)


def _ridge_thresholds(t, models, n_cols):
    """Per fold, decision thresholds midway between each class's mean training score and the
    rest's, so an unweighted fit on imbalanced classes is not tilted towards the majority."""
    out = []
    for f, (mean, weights, y_mean) in enumerate(models):
        train = np.arange(len(t.count)) != f
        class_n = t.class_count[0, train].sum(axis=0)
        class_x = t.class_sum[0, train].sum(axis=0)
        w, b = weights[:, :n_cols], y_mean[:n_cols]
        in_class = (class_x / class_n[:, None] - mean) @ w + b
        rest_x = (class_x.sum(axis=0) - class_x) / (class_n.sum() - class_n)[:, None]
        out_class = (rest_x - mean) @ w + b
        if n_cols == 1:  # binary: one column scoring class 1 against class 0
            out.append((in_class[1, 0] + in_class[0, 0]) / 2)
        else:            # a column per class: class k's rows against everyone else's
            out.append((np.diag(in_class) + np.diag(out_class)) / 2)
    return out


def _thresholded_balanced_accuracy(y_test, scores, threshold):
    if scores.shape[1] == 1:
        predicted = (scores[:, 0] >= threshold).astype(int)
    else:
        predicted = np.argmax(scores - threshold, axis=1)
    return _balanced_accuracy(y_test, predicted)


def _score_batch(n_rows, n_cols):
    """How many label sets' held-out scores fit in SCORE_BUDGET_BYTES at once."""
    return max(1, SCORE_BUDGET_BYTES // max(1, n_rows * n_cols * 8))


def _score_passes(n_rows, n_cols, n_sets):
    return -(-n_sets // _score_batch(n_rows, n_cols))


def _held_out_scores(source, t, models, n_cols, n_sets):
    """Pass 2: every row scored once, by the model of the fold that held it out. Label sets are
    scored in batches that fit SCORE_BUDGET_BYTES, so many controls cost passes, not memory.
    Yields (index of the batch's first set, rows x batch columns)."""
    batch = _score_batch(t.n_rows, n_cols)
    for first in range(0, n_sets, batch):
        last = min(n_sets, first + batch)
        cols = slice(first * n_cols, last * n_cols)
        scores = np.zeros((t.n_rows, (last - first) * n_cols))
        for positions, X in source.chunks():
            X = X - t.shift
            here = t.fold_of[positions]
            for f in np.unique(here):
                rows = here == f
                mean, weights, offset = models[f]
                scores[positions[rows]] = (X[rows] - mean) @ weights[:, cols] + offset[cols]
        yield first, scores


def _fold_aucs(label_sets, scores, splits, n_classes, model):
    """AUC per label set and fold (sets x folds) from held-out scores (rows x sets*columns): one
    column per set if binary; for several classes, LDA's discriminants become probabilities
    (softmax) and ridge's columns one-vs-rest margins, and the AUC is the one-vs-rest average
    over the classes present in the fold (as _macro_auc). All sets are ranked in one call per
    fold, which is what keeps hundreds of controls cheap."""
    n_sets, n_cols = len(label_sets), (1 if n_classes == 2 else n_classes)
    out = np.full((n_sets, len(splits)), np.nan)
    for f, (_, test) in enumerate(splits):
        y = label_sets[:, test].T.astype(int)                           # rows x sets
        s = scores[test].reshape(len(test), n_sets, n_cols)              # rows x sets x columns
        broken = np.isnan(s).any(axis=(0, 2))  # a set whose model is undefined in this fold
        if n_classes == 2:
            aucs = _batch_auc(y == 1, s[:, :, 0])
        else:
            if model == "lda":
                shifted = s - s.max(axis=2, keepdims=True)
                s = np.exp(shifted) / np.exp(shifted).sum(axis=2, keepdims=True)
            else:
                s = _ovr_margins(s)
            per_class = np.stack([_batch_auc(y == k, s[:, :, k]) for k in range(n_classes)])  # classes x sets
            n_present = (y[:, :, None] == np.arange(n_classes)).any(axis=0).sum(axis=1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # sets with a single class present
                aucs = np.where(n_present >= 2, np.nanmean(per_class, axis=0), np.nan)
        out[:, f] = np.where(broken, np.nan, aucs)
    return out


def _batch_auc(is_positive, scores):
    """_auc for every column at once: ROC-AUC from average ranks; NaN without both classes."""
    n_pos = is_positive.sum(axis=0)
    n_neg = len(is_positive) - n_pos
    ranks = rankdata(np.nan_to_num(scores), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        auc = ((ranks * is_positive).sum(axis=0) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return np.where((n_pos > 0) & (n_neg > 0), auc, np.nan)


def _ovr_margins(scores):
    """Each class's score minus the best other class's, along the last axis. One-vs-rest scores
    alone cannot single out a class lying between others (e.g. MCI between CN and AD); the
    margin, which is also what the classifier's argmax decides on, partly can."""
    order = np.argsort(scores, axis=-1)
    best = np.take_along_axis(scores, order[..., -1:], axis=-1)
    second = np.take_along_axis(scores, order[..., -2:-1], axis=-1)
    return scores - np.where(np.arange(scores.shape[-1]) == order[..., -1:], second, best)


def _probe_regression(source, y, groups, n_splits, seed, n_controls, result):
    result.update(r2_mean=float("nan"), r2_std=0.0)
    if len(y) < 10:
        result["error"] = "Too few samples for a regression probe (fewer than 10)."
        return
    if np.std(y) == 0:
        result["error"] = "The target has zero variance (all values are identical)."
        return

    if groups is None:
        n_splits = min(n_splits, len(y))
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    else:
        n_splits = min(n_splits, len(np.unique(groups)))
        cv = GroupKFold(n_splits=n_splits)
    if n_splits < 2:
        result["error"] = "Only one group: at least 2 are needed to cross-validate."
        return
    if _stream_memory_error(source.width, n_splits, "ridge", result):
        return
    result["n_splits"] = n_splits
    splits = list(cv.split(np.zeros(len(y)), y, groups))
    shuffled, shuffle_unit = _label_sets(y, groups, n_controls, seed, result["notes"], np.float32)
    n_sets = 1 + n_controls

    def targets(pos):
        """The real target in full precision, the shuffled copies from compact float32 storage."""
        return np.column_stack([y[pos], shuffled[1:, pos].T.astype(float)])

    source.ticker.plan(source.n_chunks * (1 + _score_passes(len(y), 1, n_sets)) + len(splits))
    totals = _accumulate(source, splits, targets=targets, n_targets=n_sets)
    models = _solve_ridge(totals, source.ticker)
    r2 = np.full((n_sets, len(splits)), np.nan)
    for first, scores in _held_out_scores(source, totals, models, 1, n_sets):
        for i in range(scores.shape[1]):
            p = first + i
            truth = y if p == 0 else shuffled[p].astype(float)
            r2[p] = [float(r2_score(truth[test], scores[test, i])) for _, test in splits]
    result.update(r2_mean=float(np.mean(r2[0])), r2_std=float(np.std(r2[0])), folds=r2[0].tolist())
    result.update(score_mean=result["r2_mean"], score_std=result["r2_std"])
    if n_controls:
        result["control"] = _control_summary([float(np.mean(r)) for r in r2[1:]], result["r2_mean"], shuffle_unit)


def _label_sets(y, groups, n_controls, seed, notes, dtype):
    """The real target (row 0) and `n_controls` shuffled copies (rows 1..) in one compact
    array, drawn up front so a streamed probe needs no extra pass. Returns (array, unit)."""
    sets = np.empty((1 + n_controls, len(y)), dtype=dtype)
    sets[0] = y
    if not n_controls:
        return sets, None
    rng = np.random.default_rng(seed)
    unit = _shuffle_unit(y, groups, notes)
    shuffle = _shuffler(y, groups, unit)
    for p in range(1, n_controls + 1):
        sets[p] = shuffle(rng)
    return sets, unit


def _shuffle_unit(y, groups, notes):
    """'group' if labels can be shuffled between whole groups, else 'row'."""
    if groups is None:
        return "row"
    if pd.Series(y).groupby(groups).nunique().max() == 1:
        return "group"
    notes.append("The target varies within some groups, so the control shuffles individual rows "
                 "and may understate what chance can reach.")
    return "row"


def _shuffler(y, groups, unit):
    """A function rng -> one shuffled copy of y: rows shuffled, or whole groups' labels
    shuffled between groups. The per-group setup is done once, however many copies."""
    if unit == "row":
        return lambda rng: rng.permutation(y)
    _, inverse = np.unique(groups, return_inverse=True)
    label_of = pd.Series(y).groupby(inverse).first().to_numpy()
    return lambda rng: rng.permutation(label_of)[inverse]


def _shuffle_labels(y, groups, unit, rng):
    return _shuffler(y, groups, unit)(rng)


def _control_summary(scores, real_score, unit):
    scores = np.asarray(scores, dtype=float)
    valid = scores[~np.isnan(scores)]
    at_least_real = int((valid >= real_score).sum())
    return {
        "n": int(len(scores)), "shuffled": unit,
        "mean": float(np.mean(valid)) if len(valid) else float("nan"),
        "std": float(np.std(valid)) if len(valid) else 0.0,
        "scores": scores.tolist(),
        "n_at_least_real": at_least_real,
        "p_value": (at_least_real + 1) / (len(valid) + 1),  # permutation p-value, never exactly 0
    }


# ---------------------------------------------------------------------------
# Multiple comparisons
# ---------------------------------------------------------------------------

def benjamini_hochberg(p_values) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values), in the input order.

    Calling every test with q <= alpha keeps the expected share of false discoveries among
    them at or below alpha (for independent or positively dependent tests). NaN entries
    (tests that could not be run) are left NaN and do not count towards the number of tests.
    """
    p = np.asarray(p_values, dtype=float)
    q = np.full(p.shape, np.nan)
    tested = np.flatnonzero(np.isfinite(p))
    m = len(tested)
    if not m:
        return q
    order = tested[np.argsort(p[tested], kind="stable")]
    scaled = p[order] * m / np.arange(1, m + 1)
    q[order] = np.minimum(np.minimum.accumulate(scaled[::-1])[::-1], 1.0)
    return q


def controls_for_fdr(n_tests: int, alpha: float) -> int:
    """Shuffled-label controls per probe so that a probe no shuffle matches (p = 1/(n+1)) passes a
    Benjamini-Hochberg correction across `n_tests` probes at level `alpha`, with a factor 2 to spare,
    within DEFAULT_CONTROLS..MAX_CONTROLS."""
    needed = math.ceil(2 * max(n_tests, 1) / alpha) - 1
    return int(min(max(needed, DEFAULT_CONTROLS), MAX_CONTROLS))


# ---------------------------------------------------------------------------
# Report formatter
# ---------------------------------------------------------------------------

def format_audit_report(
    title: str,
    screening: list[dict],
    performance: dict,
    probes: dict,
    thesis_ref: str = "",
) -> str:
    """Render the three audit stages into a compact Markdown summary (one slice, probes as given,
    uncorrected). prism/audit.py writes the full report for a whole cache, with the probes
    corrected for multiple comparisons."""

    lines: list[str] = []
    lines.append(f"# {title}\n")
    if thesis_ref:
        lines.append(f"_Maps to: {thesis_ref}_\n")

    # ---- Evidence ----
    lines.append("## 1. Confound & Leakage Screening\n")
    lines.append("| Field | Risk | AUC | Reason |")
    lines.append("|-------|------|-----|--------|")
    for row in screening:
        auc = "—" if row["auc"] is None else f"{row['auc']:.3f}"
        lines.append(f"| {row['field']} | **{row['risk']}** | {auc} | {row['reason']} |")
    lines.append("")

    # ---- Performance ----
    g = performance["global"]
    lines.append("## 2. Performance\n")
    lines.append(
        f"**Global** (n={g['n']}): "
        f"acc {g['accuracy']:.3f} · precision {g['precision']:.3f} · "
        f"recall {g['recall']:.3f} · F1 {g['f1']:.3f} · AUC {g['auc']:.3f}\n"
    )
    lines.append(f"### Sliced by `{performance['slice_by']}`\n")
    lines.append("| Group | n | Acc | F1 | TPR | FPR |")
    lines.append("|-------|---|-----|----|-----|-----|")
    small = False
    for name, block in performance["groups"].items():
        mark = "" if block.get("eligible", True) else "*"
        small = small or bool(mark)
        lines.append(
            f"| {name}{mark} | {block['n']} | {block['accuracy']:.3f} | "
            f"{block['f1']:.3f} | {block['tpr']:.3f} | {block['fpr']:.3f} |"
        )
    fairness = performance.get("fairness")
    gaps = fairness["gaps"] if fairness else {k: {"value": v, "ci": None} for k, v in performance["gaps"].items()}
    if not fairness:
        gaps = {name: gaps[key] for key, name in _LEGACY_GAP_NAMES}
    lines.append("")
    if small:
        lines.append(f"\\* Below the minimum group size ({fairness['min_group_size']}): shown, but not in the gaps.\n")
    if fairness and fairness["bootstrap"]["replicates"]:
        b = fairness["bootstrap"]
        lines.append(f"Intervals: {b['confidence']:.0%} bootstrap, {b['replicates']} resamples of "
                     f"{b['units']} {'subjects' if b['unit'] == 'group' else 'rows'}.\n")
    lines.append(
        f"**Fairness gaps**: ΔEO-TPR = {_gap_text(gaps['equal_opportunity'])} · "
        f"ΔEO-FPR = {_gap_text(gaps['predictive_equality'])} · "
        f"ΔEqOdds = {_gap_text(gaps['equalized_odds'])} · "
        f"ΔDemoParity = {_gap_text(gaps['demographic_parity'])}\n"
    )

    # ---- Probing ----
    lines.append("## 3. Linear Probing\n")
    lines.append("| Layer | Metric | Mean ± std | Folds | Control p |")
    lines.append("|-------|--------|------------|-------|-----------|")
    for layer_name, result in probes.items():
        metric = result.get("metric", "AUC")
        mean = result.get("score_mean", result.get("auc_mean"))
        std = result.get("score_std", result.get("auc_std"))
        folds = ", ".join(f"{a:.3f}" for a in result.get("folds", []))
        control = result.get("control")
        p_value = f"{control['p_value']:.3f}" if control else "—"
        lines.append(f"| `{layer_name}` | {metric} | {mean:.3f} ± {std:.3f} | {folds} | {p_value} |")
    lines.append("")

    return "\n".join(lines)


def _gap_text(gap):
    if not np.isfinite(gap["value"]):
        return "n/a"
    text = f"{gap['value']:.3f}"
    if gap.get("ci"):
        text += f" [{gap['ci'][0]:.3f}, {gap['ci'][1]:.3f}]"
    if gap.get("flagged"):
        text += " ⚠"
    return text
