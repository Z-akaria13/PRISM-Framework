"""
SETUP FILES
Save and reopen what the tabs were filled in with: the subject (Setup), the metadata and its
columns (Evidence), the layers and their pooling (Wiretap), and the run and cluster settings
(Execution). A setup file is a job file with two more keys (run_mode, cluster), so
`prism run --config <setup file>` runs it.
"""
import json
import os
from tkinter import filedialog, messagebox

from prism.core.dispatch import DEFAULT_CLUSTER_SETTINGS

FORMAT = 1
FILETYPES = [("PRISM setup", "*.json"), ("All files", "*.*")]


def collect(app):
    """The tabs' current settings as a setup; fields not filled in yet are left out."""
    setup, evidence, execution = app.frame_setup, app.frame_evidence, app.frame_execution
    group = evidence.cb_group.get()
    data = {
        "prism_setup": FORMAT,
        "project_root": _path(setup.ent_root.get()),
        "script_path": _path(setup.ent_script.get()),
        "framework": setup.cb_framework.get(),
        "csv_path": _path(evidence.ent_csv.get()),
        "id_col": evidence.cb_id.get(),
        "target_col": evidence.cb_target.get(),
        "group_col": None if group == "(None)" else group,
        "layers": app.frame_wiretap.extracted_layers(),
        "batch_size": _number(execution.ent_batch.get(), int),
        "output_dir": _path(execution.ent_outdir.get()),
        "device": execution.cb_device.get(),
        "max_failed_percent": _number(execution.ent_max_failed.get(), float),
        "run_mode": execution.run_mode.get(),
        "cluster": dict(execution.cluster_config),
    }
    return {key: value for key, value in data.items() if value not in (None, "")}


def save_setup(app, path=None):
    """Ask where (unless given) and write the setup there; returns the path, or None if cancelled."""
    path = path or filedialog.asksaveasfilename(title="Save Setup", defaultextension=".json", filetypes=FILETYPES)
    if not path:
        return None
    with open(path, "w", encoding="utf-8") as f:
        json.dump(collect(app), f, indent=4)
    app.lbl_setup.config(text=f"Setup: {os.path.basename(path)}")
    return path


def open_setup(app, path=None):
    """Ask which file (unless given) and fill the tabs from it. The Execution and Evidence tabs are
    filled at once; the subject is ingested in the background and its layers ticked when it has
    loaded. Returns the ingest Task, or None if there was nothing to ingest."""
    path = path or filedialog.askopenfilename(title="Open Setup", filetypes=FILETYPES)
    if not path:
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("it is not a PRISM setup")
    except (OSError, ValueError) as e:
        messagebox.showerror("Could Not Open Setup", f"{os.path.basename(path)}: {e}")
        return None

    app.lbl_setup.config(text=f"Setup: {os.path.basename(path)}")
    _apply_execution(app, data)
    problems = _apply_evidence(app, data)
    setup = app.frame_setup
    _set(setup.ent_root, data.get("project_root"))
    _set(setup.ent_script, data.get("script_path"))
    if data.get("framework") in setup.cb_framework["values"]:
        setup.cb_framework.set(data["framework"])
    if not data.get("script_path"):
        _report(problems)
        return None

    def ticked():
        missing = app.frame_wiretap.tick_layers(data.get("layers") or {})
        _report(problems + ([f"layers the model doesn't have ({', '.join(missing)})"] if missing else []))

    return setup.ingest(setup.ent_root.get(), setup.ent_script.get(), setup.cb_framework.get(), then=ticked)


def _apply_execution(app, data):
    execution = app.frame_execution
    if "batch_size" in data:
        _set(execution.ent_batch, str(data["batch_size"]))
    _set(execution.ent_outdir, data.get("output_dir"))
    if "max_failed_percent" in data:
        _set(execution.ent_max_failed, f"{float(data['max_failed_percent']):g}")
    if data.get("device"):
        execution.cb_device.set(data["device"])
    if data.get("run_mode") in ("local", "cluster"):
        execution.run_mode.set(data["run_mode"])
    if isinstance(data.get("cluster"), dict):
        execution.cluster_config.update({key: str(value) for key, value in data["cluster"].items()
                                         if key in DEFAULT_CLUSTER_SETTINGS})


def _apply_evidence(app, data):
    """Load the CSV and confirm its columns; returns what could not be restored."""
    csv = data.get("csv_path")
    if not csv:
        return []
    if not os.path.isfile(csv):
        return [f"the metadata CSV ({csv} is missing)"]
    evidence = app.frame_evidence
    _set(evidence.ent_csv, csv)
    evidence.load_csv(csv)
    columns = set(evidence.cb_id["values"])
    wanted = [data.get(key) for key in ("id_col", "target_col", "group_col")]
    missing = [col for col in wanted if col and col not in columns]
    if missing:
        return [f"columns the CSV doesn't have ({', '.join(missing)})"]
    id_col, target_col, group_col = wanted
    evidence.cb_id.set(id_col or "")
    evidence.cb_target.set(target_col or "")
    evidence.cb_group.set(group_col or "(None)")
    if id_col and target_col:
        evidence.confirm_mapping(quiet=True)
        evidence.run_leakage_check()
    return []


def _report(problems):
    if problems:
        messagebox.showwarning("Setup Partly Restored", "Not restored: " + "; ".join(problems) + ".")


def _set(entry, text):
    if text is not None:
        entry.delete(0, "end")
        entry.insert(0, str(text))


def _path(text):
    return os.path.abspath(text) if text else None


def _number(text, kind):
    try:
        return kind(text)
    except ValueError:
        return None
