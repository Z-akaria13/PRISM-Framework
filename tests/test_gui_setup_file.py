"""File > Save Setup / Open Setup: everything the tabs were filled in with, saved and restored (debt P2 #15)."""
import json

import pytest

from _support import (BLACK_BOX_CONNECTOR, CSV_NAME, H5_NAME, TORCH_CONNECTOR, ingest, make_project, run_headless,
                      set_entry, wait_for)
from prism.core.config import AuditConfig
from prism.core.dispatch import DEFAULT_CLUSTER_SETTINGS
from prism.gui.setup_file import open_setup, save_setup

pytestmark = pytest.mark.gui


def fill_in(app, tmp_path, connector=TORCH_CONNECTOR, framework="PyTorch"):
    """Fill every tab the way a user would; returns (script, csv, output dir)."""
    script, csv = make_project(tmp_path / "project", connector, n=4)
    setup = app.frame_setup
    set_entry(setup.ent_root, str(script.parent))
    set_entry(setup.ent_script, str(script))
    setup.cb_framework.set(framework)
    ingest(app, script.parent, script, framework)
    evidence = app.frame_evidence
    set_entry(evidence.ent_csv, str(csv))
    evidence.load_csv(str(csv))
    evidence.cb_id.set("subject_id")
    evidence.cb_target.set("label")
    evidence.confirm_mapping()
    wiretap = app.frame_wiretap
    items = {wiretap.tree.item(i, "text"): i for i in wiretap.tree.get_children()}
    if items:
        wiretap.tree.selection_set((items["conv"], items["fc"]))
        wiretap.set_ticked(True)
        wiretap.tree.selection_set(items["conv"])
        wiretap.set_strategy("Max Pool")
    execution = app.frame_execution
    out = tmp_path / "out"
    out.mkdir()
    set_entry(execution.ent_outdir, str(out))
    set_entry(execution.ent_batch, "3")
    set_entry(execution.ent_max_failed, "10")
    execution.cb_device.set("cpu")
    return script, csv, out


def test_a_setup_saves_every_tab_and_is_a_job_file(app, tmp_path):
    script, csv, out = fill_in(app, tmp_path)
    app.frame_execution.run_mode.set("cluster")
    app.frame_execution.cluster_config["ssh_prefix"] = "ssh me@login"

    path = save_setup(app, str(tmp_path / "my setup.json"))

    saved = json.loads((tmp_path / "my setup.json").read_text(encoding="utf-8"))
    assert saved == {
        "prism_setup": 1, "project_root": str(script.parent), "script_path": str(script), "framework": "PyTorch",
        "csv_path": str(csv), "id_col": "subject_id", "target_col": "label",
        "layers": {"conv": "Max Pool", "fc": "Mean Pool"}, "batch_size": 3, "output_dir": str(out),
        "device": "cpu", "max_failed_percent": 10.0, "run_mode": "cluster",
        "cluster": {**DEFAULT_CLUSTER_SETTINGS, "ssh_prefix": "ssh me@login"}}
    assert AuditConfig.from_dict(saved) == app.frame_execution.build_config(str(out), 3)
    assert app.lbl_setup.cget("text") == "Setup: my setup.json" and path.endswith("my setup.json")


def test_an_opened_setup_fills_every_tab_in_a_fresh_window(app, tmp_path, dialogs):
    script, csv = make_project(tmp_path / "project", TORCH_CONNECTOR, n=4)
    setup_path = tmp_path / "setup.json"
    setup_path.write_text(json.dumps({
        "prism_setup": 1, "project_root": str(script.parent), "script_path": str(script), "framework": "PyTorch",
        "csv_path": str(csv), "id_col": "subject_id", "target_col": "label",
        "layers": {"conv": "Grid Pool", "fc": "Mean Pool"}, "batch_size": 3, "output_dir": str(tmp_path / "out"),
        "device": "cpu", "max_failed_percent": 12.5, "run_mode": "cluster",
        "cluster": {"ssh_prefix": "ssh me@login", "submit_cmd": "qsub", "not_a_setting": "x"}}), encoding="utf-8")

    wait_for(app, open_setup(app, str(setup_path)))

    assert (app.project_root, app.script_path, app.framework) == (str(script.parent), str(script), "PyTorch")
    assert (app.csv_path, app.config_id_col, app.config_target_col, app.config_group_col) == (
        str(csv), "subject_id", "label", None)
    assert app.frame_wiretap.extracted_layers() == {"conv": "Grid Pool", "fc": "Mean Pool"}
    execution = app.frame_execution
    assert (execution.ent_batch.get(), execution.ent_outdir.get(), execution.cb_device.get(),
            execution.ent_max_failed.get(), execution.run_mode.get()) == ("3", str(tmp_path / "out"), "cpu", "12.5",
                                                                          "cluster")
    assert (execution.cluster_config["ssh_prefix"], execution.cluster_config["submit_cmd"]) == ("ssh me@login", "qsub")
    assert "not_a_setting" not in execution.cluster_config
    assert app.lbl_setup.cget("text") == "Setup: setup.json"
    assert dialogs.calls == []


def test_a_saved_setup_reopens_as_it_was(app, tmp_path):
    fill_in(app, tmp_path)
    path = save_setup(app, str(tmp_path / "setup.json"))
    set_entry(app.frame_execution.ent_batch, "99")
    app.frame_wiretap.tree.selection_set(app.frame_wiretap.tree.get_children())
    app.frame_wiretap.set_ticked(False)

    wait_for(app, open_setup(app, path))

    assert app.frame_execution.ent_batch.get() == "3"
    assert app.frame_wiretap.extracted_layers() == {"conv": "Max Pool", "fc": "Mean Pool"}


def test_what_cannot_be_restored_is_named_once_and_the_rest_is_restored(app, tmp_path, dialogs):
    script, csv = make_project(tmp_path / "project", TORCH_CONNECTOR, n=4)
    setup_path = tmp_path / "setup.json"
    setup_path.write_text(json.dumps({
        "project_root": str(script.parent), "script_path": str(script), "framework": "PyTorch",
        "csv_path": str(csv), "id_col": "subject_id", "target_col": "label", "group_col": "site",
        "layers": {"conv": "Max Pool", "decoder": "Mean Pool"}, "batch_size": 3}), encoding="utf-8")

    wait_for(app, open_setup(app, str(setup_path)))

    warning, = dialogs.messages("showwarning")
    assert warning == ("Not restored: columns the CSV doesn't have (site); layers the model doesn't have "
                       "(decoder).")
    assert app.frame_wiretap.extracted_layers() == {"conv": "Max Pool"}
    assert app.frame_execution.ent_batch.get() == "3"


def test_a_setup_without_a_subject_restores_the_rest(app, tmp_path, dialogs):
    setup_path = tmp_path / "setup.json"
    setup_path.write_text(json.dumps({"csv_path": str(tmp_path / "gone.csv"), "batch_size": 8, "run_mode": "cluster"}),
                          encoding="utf-8")

    assert open_setup(app, str(setup_path)) is None  # nothing to ingest

    assert dialogs.messages("showwarning") == [f"Not restored: the metadata CSV ({tmp_path / 'gone.csv'} is missing)."]
    assert (app.frame_execution.ent_batch.get(), app.frame_execution.run_mode.get()) == ("8", "cluster")


@pytest.mark.parametrize("content", ["{not json", "[1, 2]"])
def test_a_file_that_is_not_a_setup_is_refused(app, tmp_path, dialogs, content):
    path = tmp_path / "notes.json"
    path.write_text(content, encoding="utf-8")

    assert open_setup(app, str(path)) is None
    error, = dialogs.messages("showerror")
    assert error.startswith("notes.json: ")
    assert app.lbl_setup.cget("text") == ""


@pytest.mark.slow
def test_prism_run_runs_a_saved_setup(app, tmp_path):
    _, _, out = fill_in(app, tmp_path, BLACK_BOX_CONNECTOR, "Scikit-Learn")
    path = save_setup(app, str(tmp_path / "setup.json"))

    result = run_headless(path, cwd=tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert (out / H5_NAME).exists() and (out / CSV_NAME).exists()
