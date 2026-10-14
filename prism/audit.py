"""
PRISM AUDIT
Runs every analysis on one results cache and writes a Markdown report:

    python prism/audit.py <output_dir> [--slice sex site] [--concept sex site age] [--out report.md]

1. Confound screen: how well each metadata column alone predicts the target.
2. Performance and fairness gaps for each slice column, with bootstrap intervals
   (whole subjects resampled when there is a Group By column).
3. Linear probes of every layer x concept with shuffled-label controls, and a
   Benjamini-Hochberg false-discovery-rate correction across all of them, so that
   probing many layers for many concepts does not turn chance into findings.

The ID, target and Group By columns recorded with the cache are used unless overridden.
Exits 0 when the report is written, 2 when the options or the cache cannot be used.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
from dataclasses import asdict, dataclass

if __name__ == "__main__" and not __package__:
    # Run as a file (python prism/audit.py): import PRISM as the `prism` package from the folder above,
    # never this folder's modules as loose top-level names (core, gui, ...), which would load a second copy.
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from prism.analysis import (DEFAULT_BOOTSTRAP, DISPARITY_THRESHOLD, MAX_CONTROLS, MIN_GROUP_SIZE, PROBE_MODELS,
                      REGRESSION_MIN_UNIQUE, benjamini_hochberg, confound_screen, controls_for_fdr,
                      fairness_audit, linear_probe, metric_label)
from prism import __version__
from prism.core.cache import ALL_CALLS, AuditCache, CacheError
from prism.core.labels import class_order, decode_predictions

REPORT_NAME = "prism_report.md"
DEFAULT_FDR = 0.05
MAX_SLICE_VALUES = 20  # columns with more distinct values are not sliced by default

MODEL_NAMES = {"lda": "shrinkage LDA", "logistic": "logistic regression", "ridge": "ridge classifier"}
GAP_LABELS = {
    "equal_opportunity": "Equal opportunity (TPR)", "predictive_equality": "Predictive equality (FPR)",
    "demographic_parity": "Demographic parity (selection rate)", "equalized_odds": "Equalized odds (larger of TPR and FPR)",
    "accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy",
    "rmse": "RMSE", "mae": "MAE", "mean_error": "Mean error (bias)",
}


class AuditError(ValueError):
    """Options or data the audit cannot use, explained for the user."""


@dataclass
class AuditOptions:
    target: str | None = None      # default: the target recorded with the cache
    id_col: str | None = None
    group_col: str | None = None   # default: the recorded Group By column
    no_group: bool = False         # analyse rows as independent even if a Group By column was recorded
    slices: list | None = None     # default: categorical columns with 2..MAX_SLICE_VALUES values
    concepts: list | None = None   # default: the slice columns
    layers: list | None = None     # default: every stored layer (not the "<module>@all" combinations)
    positive: str | None = None    # binary targets: default "1" if present, else the last class
    model: str = PROBE_MODELS[0]
    controls: int | None = None    # default: enough for the FDR correction (controls_for_fdr)
    fdr: float = DEFAULT_FDR
    min_group_size: int = MIN_GROUP_SIZE
    bootstrap: int = DEFAULT_BOOTSTRAP
    title: str | None = None
    seed: int = 42


def run_audit(cache_dir, options: AuditOptions | None = None, log=print) -> dict:
    """Every analysis on the cache in `cache_dir`; returns the results that render_report formats."""
    options = options or AuditOptions()
    if options.model not in PROBE_MODELS:
        raise AuditError(f"Unknown probe model '{options.model}'; choose one of {', '.join(PROBE_MODELS)}.")
    if not 0 < options.fdr < 1:
        raise AuditError("The false discovery rate must be between 0 and 1.")
    cache = AuditCache.open(cache_dir)
    metadata = cache.metadata
    id_col, target, group_col = _columns(cache, options)

    reserved = {c for c in (id_col, target, group_col) if c}
    slices = options.slices if options.slices is not None else _default_slices(metadata, reserved)
    concepts = options.concepts if options.concepts is not None else list(slices)
    layers = options.layers if options.layers is not None else [
        name for name in cache.layer_names if not name.endswith(f"@{ALL_CALLS}")]
    for kind, names in (("slice", slices), ("concept", concepts)):
        missing = [c for c in names if c not in metadata.columns]
        if missing:
            raise AuditError(f"Unknown {kind} column(s): {', '.join(missing)}. Columns: {', '.join(metadata.columns)}.")
    unknown_layers = [name for name in layers if name not in cache.layer_names]
    if unknown_layers:
        raise AuditError(f"Unknown layer(s): {', '.join(unknown_layers)}. Layers: {', '.join(cache.layer_names) or 'none'}.")
    groups = metadata[group_col] if group_col else None

    log(f"Auditing {cache_dir}: {len(metadata)} rows, target '{target}'"
        + (f", Group By '{group_col}'" if group_col else ", no Group By column"))
    log("Screening metadata for confounds...")
    screening = confound_screen(metadata, target, id_cols=[id_col] if id_col else [], group_col=group_col)
    performance = _performance(cache, target, slices, groups, options, log)
    probes = _probes(cache, layers, concepts, groups, options, log)

    return {
        "title": options.title or f"PRISM audit of {os.path.basename(os.path.abspath(cache_dir))}",
        "cache": os.path.abspath(cache_dir),
        "generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "rows": int(len(metadata)),
        "rows_attempted": int(cache.rows_attempted),
        "prism": __version__,              # the version writing the report
        "provenance": cache.provenance,    # what produced the results, or None if not recorded
        "left_out": cache.failure_note,  # rows the extraction lost, which every finding below is missing
        "subjects": int(metadata[group_col].nunique()) if group_col else None,
        "columns": {"id": id_col, "target": target, "group": group_col},
        "slices": list(slices), "concepts": list(concepts), "layers": list(layers),
        "layer_shapes": {name: cache.layer_summary(name) for name in layers},  # None if not recorded
        "options": asdict(options),
        "warnings": [w for w in cache.warnings if w != cache.failure_note],
        "screening": screening,
        "performance": performance,
        "probes": probes,
    }


def _columns(cache, options):
    recorded = cache.config
    id_col = options.id_col or (recorded.id_col if recorded else None)
    target = options.target or (recorded.target_col if recorded else None)
    group_col = None if options.no_group else (options.group_col or (recorded.group_col if recorded else None))
    if not target:
        raise AuditError("This cache does not record its target column (it predates recorded settings); "
                         "name it with --target.")
    for role, col in (("target", target), ("ID", id_col), ("Group By", group_col)):
        if col and col not in cache.metadata.columns:
            raise AuditError(f"The {role} column '{col}' is not in the results' metadata.")
    return id_col, target, group_col


def _default_slices(metadata, reserved):
    return [col for col in metadata.columns
            if col not in reserved and 2 <= metadata[col].nunique() <= MAX_SLICE_VALUES]


def _performance(cache, target, slices, groups, options, log):
    y_raw = cache.metadata[target]
    raw = cache.model_output()
    notes = []
    settings = dict(groups=groups, n_boot=options.bootstrap, min_group_size=options.min_group_size, seed=options.seed)
    if pd.api.types.is_numeric_dtype(y_raw) and y_raw.nunique() > REGRESSION_MIN_UNIQUE:
        y_true, y_pred = y_raw.to_numpy(dtype=float), np.asarray(raw, dtype=float)
        settings["task"] = "regression"
    else:
        classes, notes = class_order(y_raw, cache.class_names)
        y_true = y_raw.astype(str).to_numpy()
        y_pred = decode_predictions(raw, classes, cache.output_kind)
        settings["classes"] = classes
        if len(classes) == 2:
            positive = options.positive or ("1" if "1" in classes else classes[-1])
            if positive not in classes:
                raise AuditError(f"The positive class '{positive}' is not one of {classes}.")
            settings["positive_class"] = positive
    try:
        log("Measuring performance...")
        overall = fairness_audit(y_true, y_pred, None, **settings)
        by_slice = {}
        for col in slices:
            log(f"Measuring fairness across '{col}'...")
            by_slice[col] = fairness_audit(y_true, y_pred, cache.metadata[col], **settings)
    except ValueError as e:
        raise AuditError(f"Cannot measure performance: {e}") from e
    return {"task": overall["task"], "notes": notes, "global": overall, "slices": by_slice}


def _probes(cache, layers, concepts, groups, options, log):
    planned = len(layers) * len(concepts)
    n_controls = options.controls if options.controls is not None else controls_for_fdr(planned, options.fdr)
    if not 0 <= n_controls <= MAX_CONTROLS:
        raise AuditError(f"--controls must be between 0 and {MAX_CONTROLS}.")
    rows = []
    for i, (layer, concept) in enumerate([(l, c) for l in layers for c in concepts], start=1):
        log(f"[{i}/{planned}] Probing '{layer}' for '{concept}' ({n_controls} shuffles)...")
        with cache.open_layer(layer) as features:
            result = linear_probe(features, cache.metadata[concept], groups=groups, n_controls=n_controls,
                                  model=options.model, seed=options.seed)
        rows.append({"layer": layer, "concept": concept, "result": result})

    p_values = [row["result"]["control"]["p_value"] if row["result"]["control"] and not row["result"]["error"]
                else float("nan") for row in rows]
    q_values = benjamini_hochberg(p_values)
    for row, p, q in zip(rows, p_values, q_values):
        row["p_value"], row["q_value"] = _finite(p), _finite(q)
        row["significant"] = bool(np.isfinite(q) and q <= options.fdr)
    tested = int(np.isfinite(p_values).sum()) if rows else 0

    notes = []
    if rows and n_controls == 0:
        notes.append("No shuffled-label controls were run (--controls 0), so no probe has a p-value and none "
                     "can be called significant.")
    elif tested and 1 / (n_controls + 1) > options.fdr / tested:
        notes.append(f"With {tested} probes, a probe must reach p <= {options.fdr / tested:.4f} to survive the "
                     f"correction, but {n_controls} shuffles can only show p >= {1 / (n_controls + 1):.4f}. "
                     "Probe fewer layers or concepts, or use more shuffles.")
    return {"model": options.model, "n_controls": n_controls, "fdr": options.fdr, "tested": tested,
            "rows": rows, "notes": notes}


def _finite(value):
    return float(value) if value is not None and np.isfinite(value) else None


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def render_report(audit: dict) -> str:
    out = [f"# {audit['title']}", ""]
    cols = audit["columns"]
    who = f"{audit['rows']} rows" + (f" from {audit['subjects']} subjects (Group By `{cols['group']}`)"
                                     if cols["group"] else ", each treated as independent (no Group By column)")
    out += [f"Generated {audit['generated']} from `{audit['cache']}`: {who}; target `{cols['target']}`.", ""]
    out += _summary(audit)
    out += _screening_section(audit["screening"])
    out += _performance_section(audit["performance"])
    out += _probe_section(audit["probes"])
    out += _notes_section(audit)
    out += _settings_section(audit)
    out += _provenance_section(audit)
    return "\n".join(out).rstrip() + "\n"


def _summary(audit):
    out = ["## Summary", ""]
    if audit.get("left_out"):
        out += ["**Rows left out.** " + audit["left_out"], ""]
    confounds = [r for r in audit["screening"] if r["risk"] in ("EXTREME", "HIGH")]
    out.append("**Confounds and leaks.** " + ("; ".join(
        f"`{r['field']}` alone predicts the target with AUC {r['auc']:.3f} ({r['risk']})" for r in confounds)
        + "." if confounds else "No metadata column predicts the target strongly (none HIGH or EXTREME)."))
    flagged, noise = [], 0
    for col, result in audit["performance"]["slices"].items():
        found = []
        for name, gap in result["gaps"].items():
            if name == "equalized_odds":  # repeats the larger of the TPR and FPR gaps
                continue
            if gap["flagged"]:
                label = _gap_label(name)
                found.append(f"{label[0].lower()}{label[1:]} gap {_value_ci(gap)} ({gap['high']} vs {gap['low']})")
            noise += bool(gap["noise"])
        if found:
            flagged.append(f"`{col}`: " + ", ".join(found))
    text = "; ".join(flagged) + "." if flagged else "No gap between groups is both large and clear of noise."
    if noise:
        text += f" {noise} further gap{'s' if noise != 1 else ''} above {DISPARITY_THRESHOLD} may be noise."
    out.append("")
    out.append("**Fairness.** " + text)
    probes = audit["probes"]
    found = [r for r in probes["rows"] if r["significant"]]
    out.append("")
    if not probes["rows"]:
        out.append("**Encoded concepts.** No probes were run.")
    else:
        out.append("**Encoded concepts.** " + ("; ".join(
            f"`{r['layer']}` encodes `{r['concept']}` ({r['result']['metric']} {r['result']['score_mean']:.3f}, "
            f"q = {r['q_value']:.4f})" for r in found) + "." if found else
            f"No layer encodes a probed concept beyond chance after the correction ({probes['tested']} probes, "
            f"false discovery rate {probes['fdr']})."))
    return out + [""]


def _screening_section(rows):
    out = ["## 1. Confound and leakage screen", "",
           "How well each metadata column alone predicts the target (AUC 0.5 = not at all, 1 = perfectly).", "",
           "| Column | Risk | AUC | Note |", "|---|---|---|---|"]
    order = {"EXTREME": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "NONE": 4, "N/A": 5}
    for r in sorted(rows, key=lambda r: (order.get(r["risk"], 6), -(r["auc"] or 0))):
        auc = "—" if r["auc"] is None else f"{r['auc']:.3f}"
        out.append(f"| `{r['field']}` | {r['risk']} | {auc} | {r['reason']} |")
    return out + [""]


_GROUP_COLUMNS = {
    "binary": [("Accuracy", "accuracy"), ("TPR", "tpr"), ("FPR", "fpr"), ("Selection", "selection_rate")],
    "multiclass": [("Accuracy", "accuracy"), ("Balanced acc.", "balanced_accuracy")],
    "regression": [("RMSE", "rmse"), ("MAE", "mae"), ("Mean error", "mean_error"), ("R²", "r2")],
}


def _performance_section(performance):
    task, overall = performance["task"], performance["global"]
    out = ["## 2. Performance and fairness", ""]
    b = overall["bootstrap"]
    unit = "subjects" if b["unit"] == "group" else "rows"
    if b["replicates"]:
        out += [f"Intervals are {b['confidence']:.0%} bootstrap intervals from {b['replicates']} resamples of "
                f"{b['units']} {unit}. Groups with fewer than {overall['min_group_size']} {unit} (marked \\*) "
                f"are shown but not compared. A rate gap above {DISPARITY_THRESHOLD} is flagged only if its "
                "interval excludes 0." + (" Regression gaps are flagged when their interval excludes 0."
                                         if task == "regression" else ""), ""]
    if task == "binary":
        out += [f"Positive class: `{overall['positive_class']}`.", ""]
    columns = _GROUP_COLUMNS[task]
    if task == "multiclass":
        columns = columns + [(f"Recall {c}", f"recall:{c}") for c in overall["classes"]]
    m = overall["global"]["metrics"]
    out += ["**Overall:** " + " · ".join(f"{label} {_value_ci(m[key], signed=key == 'mean_error')}"
                                          for label, key in columns), ""]
    for col, result in performance["slices"].items():
        out += [f"### By `{col}`", "", "| Group | n | Subjects | " + " | ".join(l for l, _ in columns) + " |",
                "|---|---|---|" + "---|" * len(columns)]
        for name, group in result["groups"].items():
            mark = "" if group["eligible"] else "\\*"
            subjects = group["units"] if result["bootstrap"]["unit"] == "group" else "—"
            cells = " | ".join(_value_ci(group["metrics"][key], signed=key == "mean_error") for _, key in columns)
            out.append(f"| {name}{mark} | {group['n']} | {subjects} | {cells} |")
        out += ["", "| Gap | Size [95% CI] | Between | Verdict |", "|---|---|---|---|"]
        for name, gap in result["gaps"].items():
            if gap["high"] is None:
                out.append(f"| {_gap_label(name)} | n/a | — | {gap.get('reason', 'not measurable')} |")
                continue
            verdict = "⚠ flagged" if gap["flagged"] else ("may be noise" if gap["noise"] else "—")
            out.append(f"| {_gap_label(name)} | {_value_ci(gap)} | {gap['high']} vs {gap['low']} | {verdict} |")
        out.append("")
    return out


def _probe_section(probes):
    out = ["## 3. Linear probes", ""]
    if not probes["rows"]:
        return out + ["No layers or concepts to probe.", ""]
    model = MODEL_NAMES[probes["model"]]
    out += [f"Each probe asks whether a concept can be read off a layer with a straight cut ({model}; ridge "
            f"regression for numeric concepts; cross-validated), and compares the score with {probes['n_controls']} runs on shuffled labels: p is "
            "the share of shuffles that matched it. q is p after a Benjamini-Hochberg correction across all "
            f"{probes['tested']} probes; a probe is called encoded when q ≤ {probes['fdr']}, which keeps the "
            "expected share of false findings among them at that level.", "",
            "| Layer | Concept | Score | Shuffled mean | p | q | Verdict |", "|---|---|---|---|---|---|---|"]
    for row in probes["rows"]:
        r = row["result"]
        if r["error"]:
            out.append(f"| `{row['layer']}` | `{row['concept']}` | — | — | — | — | not run: {r['error']} |")
            continue
        control = r["control"]
        control_text = f"{control['mean']:.3f}" if control else "—"
        p = "—" if row["p_value"] is None else f"{row['p_value']:.4f}"
        q = "—" if row["q_value"] is None else f"{row['q_value']:.4f}"
        verdict = "**encoded**" if row["significant"] else ("not significant" if control else "no control")
        out.append(f"| `{row['layer']}` | `{row['concept']}` | {r['metric']} {r['score_mean']:.3f} ± "
                   f"{r['score_std']:.3f} | {control_text} | {p} | {q} | {verdict} |")
    return out + [""]


def _notes_section(audit):
    notes = list(audit["warnings"]) + list(audit["performance"]["notes"]) + list(audit["probes"]["notes"])
    for col, result in audit["performance"]["slices"].items():
        notes += [f"`{col}`: {note}" for note in result["notes"]]
    notes += audit["performance"]["global"]["notes"]
    for row in audit["probes"]["rows"]:
        notes += [f"`{row['layer']}` / `{row['concept']}`: {note}" for note in row["result"]["notes"]]
    unique = list(dict.fromkeys(notes))
    if not unique:
        return []
    return ["## Notes", ""] + [f"* {note}" for note in unique] + [""]


def _settings_section(audit):
    o = audit["options"]
    return ["## Settings", "",
            f"* Slices: {', '.join(f'`{c}`' for c in audit['slices']) or 'none'}",
            f"* Concepts: {', '.join(f'`{c}`' for c in audit['concepts']) or 'none'}",
            f"* Layers: {', '.join(_layer_text(audit, c) for c in audit['layers']) or 'none'}",
            f"* Probe model {MODEL_NAMES[o['model']]}, {audit['probes']['n_controls']} shuffles per probe, false discovery "
            f"rate {o['fdr']}; {o['bootstrap']} bootstrap resamples, minimum group size {o['min_group_size']}, "
            f"seed {o['seed']}.", ""]


def _provenance_section(audit):
    out = ["## Provenance", ""]
    p = audit.get("provenance")
    if not p:
        return out + ["Not recorded: these results were extracted before PRISM recorded what produced them.", ""]
    inputs = p.get("inputs", {})

    def file_line(label, entry):
        if not entry or not entry.get("path"):
            return f"* {label}: not recorded"
        found = f"`{entry['sha256']}`" if entry.get("sha256") else "not found when the run started"
        return f"* {label}: `{entry['path']}`, {found}"

    packages = ", ".join(f"{name} {version}" for name, version in p.get("packages", {}).items())
    model = inputs.get("model")
    return out + [
        f"* Extracted with PRISM {p['prism']} on Python {p['python']} ({p['platform']}), "
        f"{p['started']} to {p['finished']}.",
        f"* Packages: {packages}.",
        file_line("Metadata CSV", inputs.get("metadata_csv")),
        file_line("Connector", inputs.get("connector")),
        f"* Model weights: `{model}`" if model else "* Model weights: no fingerprint",
        f"* Report written by PRISM {audit['prism']}.", ""]


def _layer_text(audit, name):
    summary = audit.get("layer_shapes", {}).get(name)
    return f"`{name}` ({summary})" if summary else f"`{name}`"


def _gap_label(name):
    return GAP_LABELS.get(name) or metric_label(name).capitalize()


def _value_ci(metric, signed=False, digits=3):
    value = metric["value"]
    if value is None or not np.isfinite(value):
        return "n/a"
    fmt = f"{{:+.{digits}f}}" if signed else f"{{:.{digits}f}}"
    text = fmt.format(value)
    if metric.get("ci"):
        text += f" [{fmt.format(metric['ci'][0])}, {fmt.format(metric['ci'][1])}]"
    return text


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def to_json(audit: dict) -> str:
    """The results as JSON: NaN becomes null, tuples become lists."""
    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            return float(value) if math.isfinite(value) else None
        if isinstance(value, np.ndarray):
            return clean(value.tolist())
        return value
    return json.dumps(clean(audit), indent=2)


def _emit(message):
    print(message, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="prism audit", description="Audit one PRISM results cache and write a Markdown report.")
    parser.add_argument("cache", help="the extraction's output directory")
    parser.add_argument("--out", help=f"report path (default: <cache>/{REPORT_NAME})")
    parser.add_argument("--json", help="also write the raw results as JSON to this path")
    parser.add_argument("--title")
    parser.add_argument("--target", help="target column (default: the one recorded with the cache)")
    parser.add_argument("--id", dest="id_col", help="ID column (default: recorded)")
    parser.add_argument("--group", dest="group_col", help="Group By column, e.g. subject ID (default: recorded)")
    parser.add_argument("--no-group", action="store_true", help="treat every row as independent")
    parser.add_argument("--slice", dest="slices", nargs="*", metavar="COLUMN",
                        help=f"columns to measure fairness across (default: categorical columns with 2 to "
                             f"{MAX_SLICE_VALUES} values)")
    parser.add_argument("--concept", dest="concepts", nargs="*", metavar="COLUMN",
                        help="metadata columns to probe the layers for (default: the slice columns)")
    parser.add_argument("--layer", dest="layers", nargs="*", metavar="NAME", help="layers to probe (default: all)")
    parser.add_argument("--positive", help="positive class of a binary target")
    parser.add_argument("--model", choices=PROBE_MODELS, default=PROBE_MODELS[0], help="probe model (default: lda)")
    parser.add_argument("--controls", type=int,
                        help="shuffled-label controls per probe (default: enough for the FDR correction)")
    parser.add_argument("--fdr", type=float, default=DEFAULT_FDR, help="false discovery rate (default: 0.05)")
    parser.add_argument("--min-group-size", type=int, default=MIN_GROUP_SIZE)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP, help="bootstrap resamples")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    options = AuditOptions(**{k: v for k, v in vars(args).items() if k not in ("cache", "out", "json")})
    try:
        audit = run_audit(args.cache, options, log=_emit)
    except (AuditError, CacheError, ValueError) as e:
        print(f"[ERROR] {e}", file=sys.stderr, flush=True)
        return 2
    out = args.out or os.path.join(args.cache, REPORT_NAME)
    with open(out, "w", encoding="utf-8") as f:
        f.write(render_report(audit))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            f.write(to_json(audit))
    _emit(f"[SUCCESS] Report written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
