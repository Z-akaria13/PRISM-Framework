"""core.dispatch: the job file and batch script handed to the cluster scheduler."""
import os
import shlex
import subprocess
import sys

from _support import HEADLESS, REPO_ROOT, find_bash, write_script
from prism.core.config import AuditConfig
from subprocess import CompletedProcess

import pytest

from prism.core.cancel import Cancelled
import prism.core.dispatch as dispatch
from prism.core.dispatch import (DEFAULT_CLUSTER_SETTINGS, STATUS_GRACE_SECONDS, STATUS_POLL_SECONDS, ClusterJob,
                                 SchedulerCommand, cancel_job, job_state, parse_job_id, prepare_cluster_job,
                                 render_job_script, submit_command, watch_job)


def config_for(out_dir):
    return AuditConfig(project_root="/proj", script_path="/proj/connector.py", framework="PyTorch",
                       csv_path="/proj/meta.csv", id_col="subject_id", target_col="diagnosis",
                       output_dir=str(out_dir), layers={"layer4": "Max Pool"}, batch_size=8)


def test_prepare_cluster_job_writes_the_job_and_script(tmp_path):
    config = config_for(tmp_path)

    job = prepare_cluster_job(config, DEFAULT_CLUSTER_SETTINGS, "abc123")

    assert job == ClusterJob(
        run_id="abc123",
        config_path=os.path.join(str(tmp_path), "prism_job_abc123.json"),
        script_path=os.path.join(str(tmp_path), "prism_run_abc123.sh"),
        log_path=os.path.join(str(tmp_path), "prism_log_abc123.txt"),
        submit=SchedulerCommand("sbatch prism_run_abc123.sh", shell=True, cwd=str(tmp_path)),
    )
    with open(job.config_path) as f:
        assert AuditConfig.from_json(f.read()) == config
    assert not os.path.exists(job.log_path)  # written by the compute node, not by us


def test_job_script_runs_headless_unbuffered_with_the_users_settings():
    settings = {**DEFAULT_CLUSTER_SETTINGS, "headers": "#SBATCH --gres=gpu:a100:1",
                "env_setup": "source ~/venvs/prism/bin/activate"}

    script = render_job_script(settings, "/out/job.json", "/out/log.txt")

    lines = script.splitlines()
    assert lines[0] == "#!/bin/bash"
    assert "#SBATCH --gres=gpu:a100:1" in lines
    assert "source ~/venvs/prism/bin/activate" in lines
    assert not any("PYTHONPATH" in line for line in lines)  # headless.py finds PRISM itself
    assert f"python -u {shlex.quote(str(HEADLESS))} --config /out/job.json > /out/log.txt 2>&1" in lines
    assert 'echo "[CRITICAL ERROR] Remote job died with system exit code $EXIT_CODE." >> /out/log.txt' in script


def test_job_script_has_unix_line_endings(tmp_path):
    job = prepare_cluster_job(config_for(tmp_path), DEFAULT_CLUSTER_SETTINGS, "lf0001")

    with open(job.script_path, "rb") as f:
        assert b"\r\n" not in f.read()


def test_submit_command_without_ssh_runs_in_the_output_directory():
    assert submit_command(DEFAULT_CLUSTER_SETTINGS, "/out", "/out/run.sh") == SchedulerCommand(
        "sbatch run.sh", shell=True, cwd="/out")


def test_submit_command_through_an_ssh_prefix_needs_no_local_shell():
    settings = {**DEFAULT_CLUSTER_SETTINGS, "ssh_prefix": "  ssh -p 2222 me@login.dcs  ", "submit_cmd": "qsub"}

    assert submit_command(settings, "/scratch/it's mine", "/x/run.sh") == SchedulerCommand(
        ["ssh", "-p", "2222", "me@login.dcs", "cd '/scratch/it'\"'\"'s mine' && qsub run.sh"], shell=False)


# ---------------------------------------------------------------------------
# Awkward paths (debt P1 #17): quotes, spaces, $ and backticks reach the scheduler and the node as typed
# ---------------------------------------------------------------------------

# Read by bash, $HOME and `x` would expand; by cmd.exe, %PATH% would. A double quote is not
# allowed in a Windows file name; everything else here is.
AWKWARD = "it's $HOME `x` %PATH% & y" + ("" if os.name == "nt" else ' "q"')

FAKE_SBATCH = """
import os
import sys

# Records where it ran and what it was given, then answers like sbatch.
with open(os.environ["PRISM_TEST_RECORD"], "w", encoding="utf-8") as f:
    f.write(os.getcwd() + "\\n" + sys.argv[1])
print("Submitted batch job 4242")
"""

FAKE_SSH = """
import subprocess
import sys

# Like ssh: the last argument is the command, which the login shell on the host reads.
host, command = sys.argv[1], sys.argv[2:]
assert len(command) == 1, command
sys.exit(subprocess.run([{bash!r}, "-c", command[0]]).returncode)
"""


@pytest.fixture
def awkward_dir(tmp_path):
    path = tmp_path / AWKWARD
    path.mkdir()
    return path


def needs_bash():
    bash = find_bash()
    if bash is None:
        pytest.skip("needs bash (Git Bash on Windows)")
    return bash


def submit_and_record(tmp_path, monkeypatch, out, settings):
    record = tmp_path / "record.txt"
    monkeypatch.setenv("PRISM_TEST_RECORD", str(record))
    job = prepare_cluster_job(config_for(out), settings, "abc123")
    result = job.submit.run(capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return result, record.read_text(encoding="utf-8").splitlines()


def test_a_local_submit_runs_in_an_awkward_output_directory(tmp_path, monkeypatch, awkward_dir):
    sbatch = write_script(tmp_path / "fake_sbatch.py", FAKE_SBATCH)
    settings = {**DEFAULT_CLUSTER_SETTINGS, "submit_cmd": f'"{sys.executable}" "{sbatch}"'}

    result, (cwd, script) = submit_and_record(tmp_path, monkeypatch, awkward_dir, settings)

    assert parse_job_id(result.stdout) == "4242"
    assert os.path.samefile(cwd, awkward_dir) and script == "prism_run_abc123.sh"


@pytest.mark.slow
def test_a_submit_through_ssh_reaches_an_awkward_output_directory(tmp_path, monkeypatch, awkward_dir):
    bash = needs_bash()
    sbatch = write_script(tmp_path / "fake_sbatch.py", FAKE_SBATCH)
    ssh = write_script(tmp_path / "fake_ssh.py", FAKE_SSH.format(bash=bash))
    python = sys.executable.replace("\\", "/")  # as the login shell would see it
    settings = {**DEFAULT_CLUSTER_SETTINGS, "ssh_prefix": f'"{sys.executable}" "{ssh}" me@login',
                "submit_cmd": f'"{python}" "{str(sbatch).replace(os.sep, "/")}"'}

    result, (cwd, script) = submit_and_record(tmp_path, monkeypatch, awkward_dir, settings)

    assert parse_job_id(result.stdout) == "4242"
    assert os.path.samefile(cwd, awkward_dir) and script == "prism_run_abc123.sh"


@pytest.mark.slow
def test_the_job_script_hands_awkward_paths_to_python_as_typed(tmp_path, monkeypatch, awkward_dir):
    bash = needs_bash()
    # Stand-in for headless.py, in the awkward directory too: prints the arguments it was given.
    worker = write_script(awkward_dir / "worker.py", "import sys\nprint(sys.argv[1:])\n")
    monkeypatch.setattr(dispatch, "HEADLESS_SCRIPT", str(worker))
    settings = {**DEFAULT_CLUSTER_SETTINGS, "headers": "", "env_setup": f'python() {{ "{sys.executable}" "$@"; }}'}
    job = prepare_cluster_job(config_for(awkward_dir), settings, "abc123")

    subprocess.run([bash, job.script_path], check=True, timeout=60)

    with open(job.log_path, encoding="utf-8") as f:
        assert f.read().strip() == str(["--config", job.config_path])


@pytest.mark.parametrize("output", ["Job <4242> is submitted to queue <normal>.", "4242; rm -rf ~", "`reboot`"])
def test_output_that_does_not_end_in_a_job_id_gives_none(output):
    assert parse_job_id(output) is None


def test_status_and_cancel_need_a_usable_job_id():
    never = lambda cmd, **kw: pytest.fail("the scheduler was asked about an unusable job ID")  # noqa: E731

    assert job_state(DEFAULT_CLUSTER_SETTINGS, "4242; rm -rf ~", run=never) is None
    assert "no job ID" in cancel_job(DEFAULT_CLUSTER_SETTINGS, None, run=never)


# ---------------------------------------------------------------------------
# Following a submitted job: its log and the scheduler's queue
# ---------------------------------------------------------------------------

class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class Scheduler:
    """Stands in for `run` on status and cancel commands: lists the job until `gone_at` (clock time)."""

    def __init__(self, clock, gone_at=None, broken=False):
        self.clock, self.gone_at, self.broken, self.commands = clock, gone_at, broken, []

    def __call__(self, cmd, **kwargs):
        self.commands.append(cmd)
        if self.broken:
            return CompletedProcess(cmd, 127, "", "bash: squeue: command not found")
        listed = self.gone_at is None or self.clock.now < self.gone_at
        return CompletedProcess(cmd, 0, "4242 gpu prism R 0:05" if listed else "", "")


def watch(tmp_path, writes, settings=DEFAULT_CLUSTER_SETTINGS, cancel_at=None, **scheduler):
    """Follow a fake job whose log gets `writes` ({clock time: text appended then}); time only
    passes when the watcher sleeps. Returns (outcome, log lines, progress, statuses, clock, scheduler)."""
    log_path = tmp_path / "job.log"
    clock = Clock()
    pending = sorted(writes.items())
    lines, progress, statuses = [], [], []
    run = Scheduler(clock, **scheduler)

    def sleep(seconds):
        clock.now += seconds
        while pending and pending[0][0] <= clock.now:
            with open(log_path, "a") as f:
                f.write(pending.pop(0)[1])
        if cancel_at is not None and clock.now >= cancel_at:
            raise Cancelled()
        assert clock.now < 10_000, "the watch never ended"

    outcome = watch_job(str(log_path), "4242", settings, log=lines.append, progress=progress.append,
                        status=statuses.append, sleep=sleep, run=run, clock=clock)
    return outcome, lines, progress, statuses, clock, run


def test_a_finished_job_is_complete(tmp_path):
    outcome, lines, progress, statuses, _, _ = watch(tmp_path, {
        5: "Targeting layers: []\n[PROGRESS] 50.0\n", 10: "[SUCCESS] Extracted 5/5 items\n"})

    assert outcome == "complete"
    assert lines == ["Targeting layers: []", "[SUCCESS] Extracted 5/5 items"]
    assert progress == [50.0]
    assert statuses[0] == "Waiting in queue... (0 s)" and statuses[-1] == "Executing on compute node..."


def test_partly_written_lines_are_held_back(tmp_path):
    _, lines, progress, _, _, _ = watch(tmp_path, {5: "[PROGRESS] 2", 6: "5.0\nHello wor",
                                                   7: "ld\n[SUCCESS] done\n"})

    assert progress == [25.0]
    assert lines == ["Hello world", "[SUCCESS] done"]


def test_lines_written_just_after_a_failure_are_still_shown(tmp_path):
    outcome, lines, _, _, _, _ = watch(tmp_path, {
        5: "[CRITICAL ERROR] CUDA out of memory\n", 5.5: "Traceback (most recent call last): ...\n",
        6.5: "[CRITICAL ERROR] Remote job died with system exit code 1.\n"})

    assert outcome == "failed"
    assert lines == ["[CRITICAL ERROR] CUDA out of memory", "Traceback (most recent call last): ...",
                     "[CRITICAL ERROR] Remote job died with system exit code 1."]


def test_a_job_killed_by_the_scheduler_is_reported_as_lost(tmp_path):
    outcome, lines, _, _, clock, run = watch(tmp_path, {5: "Targeting layers: ['fc']\n[PROGRESS] 25.0\n"},
                                             gone_at=100)

    assert outcome == "lost"
    assert "scheduler no longer lists job 4242" in lines[-1] and "time or memory limit" in lines[-1]
    first_check_after = -(-100 // STATUS_POLL_SECONDS) * STATUS_POLL_SECONDS
    assert clock.now == pytest.approx(first_check_after + STATUS_GRACE_SECONDS, abs=1.5)
    assert run.commands[0] == "squeue -h -j 4242"


def test_a_job_that_leaves_the_queue_without_ever_starting_is_lost(tmp_path):
    outcome, _, _, statuses, _, _ = watch(tmp_path, {}, gone_at=40)

    assert outcome == "lost"
    assert all(s.startswith("Waiting in queue") for s in statuses)


def test_a_log_that_catches_up_within_the_grace_period_wins(tmp_path):
    # gone at the first check (30 s); the log's verdict reaches the shared disk at 45 s
    outcome, _, _, _, _, _ = watch(tmp_path, {5: "Targeting layers: []\n", 45: "[SUCCESS] Extracted 5/5 items\n"},
                                   gone_at=29)

    assert outcome == "complete"


def test_an_unusable_status_command_is_reported_once_and_the_watch_goes_on(tmp_path):
    outcome, lines, _, _, _, run = watch(tmp_path, {200: "[SUCCESS] Extracted 5/5 items\n"}, broken=True)

    assert outcome == "complete"
    assert len(run.commands) > 1 and sum("Could not ask the scheduler" in line for line in lines) == 1


def test_without_a_status_command_the_scheduler_is_never_asked(tmp_path):
    settings = dict(DEFAULT_CLUSTER_SETTINGS, status_cmd="")

    outcome, _, _, _, _, run = watch(tmp_path, {200: "[SUCCESS] Extracted 5/5 items\n"}, settings=settings)

    assert outcome == "complete" and run.commands == []


def test_cancelling_stops_the_watch_and_closes_the_log(tmp_path):
    with pytest.raises(Cancelled):
        watch(tmp_path, {1: "Targeting layers: []\n"}, cancel_at=3)

    (tmp_path / "job.log").unlink()  # fails on Windows if the watcher still had it open


@pytest.mark.parametrize("output, job_id", [
    ("Submitted batch job 4242\n", "4242"), ("12345.pbs01\n", "12345.pbs01"), ("", None)])
def test_the_job_id_is_the_last_word_the_scheduler_printed(output, job_id):
    assert parse_job_id(output) == job_id


@pytest.mark.parametrize("result, state", [
    (CompletedProcess("", 0, "4242 gpu prism R", ""), True),
    (CompletedProcess("", 0, "", ""), False),  # finished: squeue lists nothing
    (CompletedProcess("", 1, "", "slurm_load_jobs error: Invalid job id specified"), False),  # purged
    (CompletedProcess("", 255, "", "ssh: connect to host login: Connection refused"), None),
])
def test_job_state_reads_the_status_command(result, state):
    commands = []
    settings = dict(DEFAULT_CLUSTER_SETTINGS, ssh_prefix="ssh login")

    assert job_state(settings, "4242", run=lambda cmd, **kw: commands.append((cmd, kw["shell"])) or result) is state
    assert commands == [(["ssh", "login", "squeue -h -j 4242"], False)]


def test_cancel_job_runs_the_cancel_command():
    commands = []
    ok = cancel_job(DEFAULT_CLUSTER_SETTINGS, "4242",
                    run=lambda cmd, **kw: commands.append(cmd) or CompletedProcess(cmd, 0, "", ""))
    refused = cancel_job(DEFAULT_CLUSTER_SETTINGS, "4242",
                         run=lambda cmd, **kw: CompletedProcess(cmd, 1, "", "scancel: Access denied"))

    assert commands == ["scancel 4242"] and ok == "[CLUSTER] Cancelled job 4242."
    assert "Access denied" in refused
    assert "keeps running" in cancel_job(dict(DEFAULT_CLUSTER_SETTINGS, cancel_cmd=""), "4242")
