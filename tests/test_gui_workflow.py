"""End-to-end: drive the tabs in order, and check the GUI -> headless cluster job contract."""
import json
from subprocess import CompletedProcess
from types import SimpleNamespace

import pytest

from _support import (BLACK_BOX_CONNECTOR, CSV_NAME, H5_NAME, ValueConnector, ingest, make_project, run_engine,
                      run_headless, run_mainloop_until, set_entry, value_frame, wait_for, wait_idle)
from prism.core.config import AuditConfig

pytestmark = pytest.mark.gui


def ingest_and_map(app, tmp_path, n=8):
    """Setup tab -> Evidence tab, the way a user would fill them in."""
    script, csv = make_project(tmp_path / "project", BLACK_BOX_CONNECTOR, n=n)
    setup = app.frame_setup
    set_entry(setup.ent_root, str(script.parent))
    set_entry(setup.ent_script, str(script))
    setup.cb_framework.set("Scikit-Learn")
    ingest(app, setup.ent_root.get(), setup.ent_script.get(), setup.cb_framework.get())

    evidence = app.frame_evidence
    set_entry(evidence.ent_csv, str(csv))
    evidence.load_csv(str(csv))
    evidence.cb_id.set("subject_id")
    evidence.cb_target.set("label")
    evidence.confirm_mapping()
    app.update()


def test_black_box_audit_from_setup_to_gallery(app, tmp_path):
    ingest_and_map(app, tmp_path)
    execution = app.frame_execution
    out = tmp_path / "out"
    out.mkdir()
    set_entry(execution.ent_outdir, str(out))
    set_entry(execution.ent_batch, "3")

    execution.start_execution()
    run_mainloop_until(app, lambda: str(execution.btn_run.cget("state")) == "normal")

    assert execution.lbl_status.cget("text") == "Extraction Complete!"
    assert app.active_cache_dir == str(out)
    assert (out / H5_NAME).exists() and (out / CSV_NAME).exists()
    console = execution.txt_console.get("1.0", "end")
    assert "[SUCCESS] Extracted 8/8" in console
    assert execution.progress_var.get() == pytest.approx(100.0)

    app.frame_performance.load_results()
    wait_idle(app, app.frame_performance.runner)
    assert app.frame_performance.lbl_stats.cget("text").startswith("Global Acc:")

    app.frame_gallery.load_data()
    assert len(app.frame_gallery.tree.get_children()) == 8


def test_a_run_that_left_rows_out_warns_and_every_analysis_tab_says_so(app, dialogs, tmp_path, monkeypatch):
    ingest_and_map(app, tmp_path)
    execution = app.frame_execution
    load_item = app.connector_instance.load_item

    def fails_from_row_5(row_id, meta):
        if meta["value"] >= 5:
            raise OSError(f"cannot read {row_id}")
        return load_item(row_id, meta)

    monkeypatch.setattr(app.connector_instance, "load_item", fails_from_row_5)
    out = tmp_path / "out"
    out.mkdir()
    set_entry(execution.ent_outdir, str(out))
    set_entry(execution.ent_max_failed, "50")

    execution.start_execution()
    run_mainloop_until(app, lambda: str(execution.btn_run.cget("state")) == "normal")

    assert execution.lbl_status.cget("text") == "Extraction complete: 3 of 8 rows were left out. Check console."
    note, = dialogs.messages("showwarning")
    assert note.startswith("3 of 8 rows (38%) failed during the extraction and are not in these results: "
                           "3 in load_item (first error: OSError: cannot read S005).")

    app.frame_performance.load_results()
    wait_idle(app, app.frame_performance.runner)
    assert note in app.frame_performance.txt_report.get("1.0", "end")
    app.frame_probing.load_options()
    assert note in app.frame_probing.txt_res.get("1.0", "end")
    app.frame_gallery.load_data()
    assert note in app.frame_gallery.lbl_notes.cget("text")


def test_a_run_stopped_by_the_failure_limit_says_why(app, dialogs, tmp_path, monkeypatch):
    ingest_and_map(app, tmp_path)
    execution = app.frame_execution
    monkeypatch.setattr(app.connector_instance, "load_item", lambda row_id, meta: 1 / 0)
    out = tmp_path / "out"
    out.mkdir()
    set_entry(execution.ent_outdir, str(out))

    execution.start_execution()
    run_mainloop_until(app, lambda: str(execution.btn_run.cget("state")) == "normal")

    assert execution.lbl_status.cget("text") == "Extraction Failed. Check console."
    console = execution.txt_console.get("1.0", "end")
    assert "[CRITICAL ERROR] Stopped: more than 5% of the 8 rows failed" in console
    assert "ZeroDivisionError: division by zero" in console
    assert app.active_cache_dir != str(out) and list(out.iterdir()) == []


@pytest.mark.parametrize("text", ["five", "-1", "101"])
def test_an_unusable_failure_limit_is_refused(app, dialogs, tmp_path, text):
    ingest_and_map(app, tmp_path)
    execution = app.frame_execution
    set_entry(execution.ent_outdir, str(tmp_path))
    set_entry(execution.ent_max_failed, text)

    execution.start_execution()

    error, = dialogs.messages("showerror")
    assert "Invalid execution settings" in error and ("percentage from 0 to 100" in error
                                                      or "max_failed_percent must be a number from 0 to 100" in error)


def test_the_execution_tab_builds_the_job_from_every_tab(app, tmp_path):
    ingest_and_map(app, tmp_path, n=5)
    app.frame_execution.cb_device.set("cpu")

    config = app.frame_execution.build_config(str(tmp_path / "out"), 4)

    project = tmp_path / "project"
    assert config.project_root == str(project)
    assert config.script_path == str(project / "connector.py")
    assert config.csv_path == str(project / "meta.csv")
    assert (config.framework, config.id_col, config.target_col, config.group_col) == (
        "Scikit-Learn", "subject_id", "label", None)
    assert config.layers == {}
    assert (config.batch_size, config.device, config.output_dir) == (4, "cpu", str(tmp_path / "out"))
    assert config.max_failed_percent == 5.0


def remote_job(app, tmp_path, monkeypatch, log_text, results=False):
    """Submit through the Execution tab with a fake scheduler whose job has already written `log_text`
    (and, with `results`, its results)."""
    from prism.inspectors.sklearn import SklearnInspector
    import prism.gui.frames.execution as execution_module
    ingest_and_map(app, tmp_path, n=5)
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(execution_module, "uuid", SimpleNamespace(uuid4=lambda: "abc123-fixed"))
    monkeypatch.setattr(execution_module, "subprocess", SimpleNamespace(
        run=lambda cmd, **kwargs: CompletedProcess(cmd, 0, stdout="Submitted batch job 42", stderr="")))
    if results:
        assert run_engine(out, value_frame(5), ValueConnector(), SklearnInspector(None)).ok
    (out / "prism_log_abc123.txt").write_text(log_text)

    execution = app.frame_execution
    wait_for(app, execution.run_cluster(execution.build_config(str(out), 4)))
    return execution, out


def test_cluster_monitor_relays_the_remote_log(app, tmp_path, monkeypatch):
    execution, out = remote_job(app, tmp_path, monkeypatch,
                                "Targeting layers: []\n[PROGRESS] 50.0\n\n[SUCCESS] Extracted 5/5 items\n",
                                results=True)

    console = execution.txt_console.get("1.0", "end")
    assert "[CLUSTER] Job Accepted: Submitted batch job 42" in console
    assert "Targeting layers: []" in console and "[PROGRESS]" not in console
    assert "[SUCCESS] Remote Extraction Complete." in console
    assert execution.progress_var.get() == pytest.approx(50.0)
    assert execution.lbl_status.cget("text") == "Complete!"
    assert app.active_cache_dir == str(out)
    assert str(execution.btn_run.cget("state")) == "normal"


def test_a_remote_success_whose_results_cannot_be_seen_says_so(app, tmp_path, monkeypatch):
    # e.g. the output directory is not on a filesystem shared with the cluster
    execution, out = remote_job(app, tmp_path, monkeypatch, "[SUCCESS] Extracted 5/5 items\n")

    assert execution.lbl_status.cget("text") == "Extraction finished, but its results can't be opened. Check console."
    console = execution.txt_console.get("1.0", "end")
    assert "[ERROR] The results can't be opened: Cache files not found" in console
    assert "a filesystem this computer shares with the cluster" in console


def test_cluster_monitor_reports_a_failed_remote_job_as_failed(app, tmp_path, monkeypatch):
    execution, _ = remote_job(app, tmp_path, monkeypatch,
                              "Targeting layers: []\n[CRITICAL ERROR] CUDA out of memory\n"
                              "[CRITICAL ERROR] Remote job died with system exit code 1.\n")

    console = execution.txt_console.get("1.0", "end")
    assert "[CRITICAL ERROR] CUDA out of memory" in console
    assert "Remote extraction failed" in console
    assert "Remote Extraction Complete" not in console
    assert execution.lbl_status.cget("text") == "Extraction Failed. Check console."
    assert app.active_cache_dir is None
    assert str(execution.btn_run.cget("state")) == "normal"


@pytest.mark.slow
def test_cluster_job_written_by_the_gui_runs_in_headless(app, tmp_path, monkeypatch):
    import prism.gui.frames.execution as execution_module
    ingest_and_map(app, tmp_path, n=5)
    out = tmp_path / "out"
    out.mkdir()
    submitted = []

    def fake_run(cmd, **kwargs):
        submitted.append(cmd)
        return CompletedProcess(cmd, 1, stdout="", stderr="sbatch: command not found")

    monkeypatch.setattr(execution_module, "subprocess", SimpleNamespace(run=fake_run))
    config = app.frame_execution.build_config(str(out), 4)

    wait_for(app, app.frame_execution.run_cluster(config))

    job_file = next(out.glob("prism_job_*.json"))
    script = next(out.glob("prism_run_*.sh")).read_text()
    job = json.loads(job_file.read_text())
    assert job["framework"] == "Scikit-Learn" and job["layers"] == {} and job["batch_size"] == 4
    assert AuditConfig.from_json(job_file.read_text()) == config
    assert "#SBATCH --partition=gpu" in script and "python -u" in script and job_file.name in script
    assert "sbatch prism_run_" in submitted[0]
    assert "[ERROR] Submit failed: sbatch: command not found" in app.frame_execution.txt_console.get("1.0", "end")

    result = run_headless(job_file, cwd=tmp_path)

    assert result.returncode == 0, result.stderr
    assert "[SUCCESS] Extracted 5/5" in result.stdout


def test_existing_results_are_only_replaced_after_asking(app, tmp_path, dialogs):
    ingest_and_map(app, tmp_path)
    execution = app.frame_execution
    out = tmp_path / "out"
    out.mkdir()
    (out / H5_NAME).write_text("results from an earlier run")
    set_entry(execution.ent_outdir, str(out))

    dialogs.answer = False
    execution.start_execution()

    assert dialogs.kinds[-1] == "askyesno"
    assert str(execution.btn_run.cget("state")) != "disabled"
    assert (out / H5_NAME).read_text() == "results from an earlier run"

    dialogs.answer = True
    execution.start_execution()
    run_mainloop_until(app, lambda: str(execution.btn_run.cget("state")) == "normal")

    assert execution.lbl_status.cget("text") == "Extraction Complete!"



# ---------------------------------------------------------------------------
# Background work: cancelling, killed cluster jobs
# ---------------------------------------------------------------------------

def test_a_local_extraction_can_be_cancelled(app, tmp_path):
    import threading
    ingest_and_map(app, tmp_path, n=8)
    execution = app.frame_execution
    loading, release = threading.Event(), threading.Event()
    load_item = app.connector_instance.load_item

    def slow_load(row_id, meta):
        loading.set()
        release.wait(5)  # hold the first item until Cancel has been pressed
        return load_item(row_id, meta)

    app.connector_instance.load_item = slow_load
    out = tmp_path / "out"
    out.mkdir()
    set_entry(execution.ent_outdir, str(out))

    execution.start_execution()
    assert loading.wait(5)
    assert str(execution.btn_cancel.cget("state")) == "normal"
    execution.cancel_execution()
    release.set()
    wait_for(app, execution.task)

    assert execution.lbl_status.cget("text") == "Extraction cancelled; nothing was saved."
    assert "[CANCELLED]" in execution.txt_console.get("1.0", "end")
    assert not (out / H5_NAME).exists() and app.active_cache_dir is None
    assert str(execution.btn_run.cget("state")) == "normal"
    assert str(execution.btn_cancel.cget("state")) == "disabled"


def fast_watch(monkeypatch):
    import prism.core.dispatch as dispatch
    monkeypatch.setattr(dispatch, "LOG_POLL_SECONDS", 0.01)
    monkeypatch.setattr(dispatch, "STATUS_POLL_SECONDS", 0.05)
    monkeypatch.setattr(dispatch, "STATUS_GRACE_SECONDS", 0.1)


def scheduler(commands, listed):
    """A fake subprocess.run: accepts the submission, then lists the job while `listed()` is true."""
    def run(cmd, **kwargs):
        commands.append(cmd)
        if "squeue" in cmd:
            return CompletedProcess(cmd, 0, "42 gpu prism R" if listed() else "", "")
        if "scancel" in cmd:
            return CompletedProcess(cmd, 0, "", "")
        return CompletedProcess(cmd, 0, stdout="Submitted batch job 42", stderr="")
    return run


def test_a_cluster_job_killed_by_the_scheduler_is_reported(app, tmp_path, monkeypatch):
    import prism.gui.frames.execution as execution_module
    fast_watch(monkeypatch)
    ingest_and_map(app, tmp_path, n=5)
    out = tmp_path / "out"
    out.mkdir()
    commands = []
    monkeypatch.setattr(execution_module, "uuid", SimpleNamespace(uuid4=lambda: "abc123-fixed"))
    monkeypatch.setattr(execution_module, "subprocess", SimpleNamespace(run=scheduler(commands, lambda: False)))
    (out / "prism_log_abc123.txt").write_text("Targeting layers: []\n[PROGRESS] 40.0\n")  # then killed

    execution = app.frame_execution
    wait_for(app, execution.run_cluster(execution.build_config(str(out), 4)))

    console = execution.txt_console.get("1.0", "end")
    assert "scheduler no longer lists job 42" in console
    assert execution.lbl_status.cget("text") == "The job ended without finishing. Check console."
    assert any("squeue -h -j 42" in cmd for cmd in commands)
    assert app.active_cache_dir is None


def test_cancelling_a_cluster_watch_can_cancel_the_job(app, tmp_path, monkeypatch, dialogs):
    import prism.gui.frames.execution as execution_module
    fast_watch(monkeypatch)
    ingest_and_map(app, tmp_path, n=5)
    out = tmp_path / "out"
    out.mkdir()
    commands = []
    monkeypatch.setattr(execution_module, "uuid", SimpleNamespace(uuid4=lambda: "abc123-fixed"))
    monkeypatch.setattr(execution_module, "subprocess", SimpleNamespace(run=scheduler(commands, lambda: True)))
    execution = app.frame_execution
    task = execution.run_cluster(execution.build_config(str(out), 4))
    while execution._cluster_job is None:  # wait until the job has been submitted
        app.update()

    dialogs.answer = True
    execution.cancel_execution()
    wait_for(app, task)

    assert dialogs.kinds[-1] == "askyesno" and "job 42" in dialogs.calls[-1][2]
    assert "scancel 42" in commands
    assert "[CLUSTER] Cancelled job 42." in execution.txt_console.get("1.0", "end")
    assert execution.lbl_status.cget("text") == "Stopped watching the cluster job."
