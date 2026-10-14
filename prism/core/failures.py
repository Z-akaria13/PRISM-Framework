"""
FAILED ROWS
Rows an extraction could not use: an item load_item could not load, a batch run_inference
crashed on, or a batch whose outputs did not fit the others. They are left out of the results,
listed in prism_failed_rows.csv, and described in plain language wherever the results are used.

Failures are rarely random: one site's files in another format fail as a group. So the
description names the groups that lost noticeably more of their rows than the rest.
"""
import pandas as pd

FAILED_AT = "prism_failed_at"  # columns added to each failed row: where it failed ...
ERROR = "prism_error"          # ... and why
STAGES = ("load_item", "run_inference", "saving")
MAX_GROUPS = 20      # columns with more distinct values (ages, IDs) are not treated as groups
MAX_AFFECTED = 3     # groups named in a description
ERROR_CHARS = 300


def error_text(error):
    """One line for an exception: its type and message, shortened."""
    text = " ".join(f"{type(error).__name__}: {error}".split())
    return text if len(text) <= ERROR_CHARS else text[:ERROR_CHARS - 3] + "..."


def percent(part, whole):
    share = 100 * part / whole if whole else 0.0
    return f"{share:.1f}%" if share < 10 else f"{share:.0f}%"


def by_stage(failed):
    """'30 in load_item (first error: OSError: ...), 4 in run_inference (...)' for the failed rows."""
    parts = []
    stages = failed[FAILED_AT].astype(str)
    order = [s for s in STAGES if s in set(stages)] + sorted(set(stages) - set(STAGES))
    for stage in order:
        rows = failed[stages == stage]
        where = "while saving" if stage == "saving" else f"in {stage}"
        parts.append(f"{len(rows)} {where} (first error: {rows[ERROR].iloc[0]})")
    return ", ".join(parts)


def most_affected(attempted, failed_mask, exclude=()):
    """The groups (column = value) that lost the largest share of their rows, if noticeably more than
    the rest: at least 2 failed rows, and every row of the group or at least twice the overall rate.

    attempted: every row tried (metadata columns); failed_mask: one bool per row of `attempted`."""
    attempted = attempted.reset_index(drop=True)
    failed_mask = pd.Series(list(failed_mask), dtype=bool)
    overall = failed_mask.mean() if len(failed_mask) else 0.0
    if overall >= 1:
        return []  # everything failed: no group stands out
    found = []
    for col in attempted.columns:
        if col in exclude or col in (FAILED_AT, ERROR):
            continue
        values = attempted[col]
        if not 2 <= values.nunique(dropna=True) <= MAX_GROUPS:
            continue
        for value, rows in failed_mask.groupby(values, dropna=True):
            n, f = len(rows), int(rows.sum())
            if f >= 2 and (f == n or f / n >= 2 * overall):
                found.append((f / n, f, n, col, value))
    found.sort(key=lambda g: (-g[0], -g[1]))
    return [(col, value, f, n) for _, f, n, col, value in found[:MAX_AFFECTED]]


def describe_groups(groups):
    """'site = B (all 30 rows: missing from every analysis), scanner = GE (30 of 45 rows)'."""
    return ", ".join(f"{col} = {value} (" + (f"all {n} rows: missing from every analysis" if f == n
                                              else f"{f} of {n} rows") + ")"
                     for col, value, f, n in groups)


def describe(attempted, failed_mask, failed, exclude=()):
    """(failures by stage, most affected groups or None).

    attempted: every row tried, failed ones included; failed_mask: which of them failed;
    failed: the failed rows with their FAILED_AT and ERROR columns."""
    groups = most_affected(attempted, failed_mask, exclude)
    return by_stage(failed), (describe_groups(groups) if groups else None)