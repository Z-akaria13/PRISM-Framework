"""
CLUSTER DISPATCH
Turns an AuditConfig into a job file plus a batch script that runs headless.py
on a compute node, and follows the job (its log and the scheduler's queue) once it is
submitted. Nothing here touches Tk; the Execution tab submits and monitors.

Quoting: the batch script and the cluster's login shell are POSIX shells, so every path PRISM puts
into them is quoted with shlex.quote. Through SSH, ssh is started directly with the remote command
as one argument, so no local shell (cmd.exe on Windows) reads it. Without SSH, the submit command
runs in the output directory instead of cd-ing there. The commands typed into Cluster Settings
are used as written.
"""
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass

PRISM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HEADLESS_SCRIPT = os.path.join(PRISM_DIR, "headless.py")

DEFAULT_CLUSTER_SETTINGS = {
    "ssh_prefix": "",
    "submit_cmd": "sbatch",
    "headers": "#SBATCH --partition=gpu\n#SBATCH --gres=gpu:1\n#SBATCH --mem=32G\n#SBATCH --time=02:00:00",
    "env_setup": "module load CUDA",
    # {job_id} is the last word the submit command printed ("Submitted batch job 4242" -> 4242).
    # Prints the job while the scheduler still knows it; empty once it has left the queue. Blank: never asked.
    "status_cmd": "squeue -h -j {job_id}",
    "cancel_cmd": "scancel {job_id}",
}

LOG_POLL_SECONDS = 1.0     # how often the job's log is read for new lines
STATUS_POLL_SECONDS = 30.0  # how often the scheduler is asked whether the job still exists
STATUS_GRACE_SECONDS = 30.0  # once it has left the queue, how long to wait for the log to say how it ended


JOB_ID = re.compile(r"[\w.\-]+")  # what PRISM accepts as a job ID: e.g. 4242, 12345.pbs01


@dataclass(frozen=True)
class SchedulerCommand:
    """One command for the scheduler, ready for subprocess.run: ssh's arguments (`shell` False), or
    a line for the local shell, run in `cwd`."""
    args: object
    shell: bool
    cwd: str | None = None

    def run(self, run=subprocess.run, **kwargs):
        return run(self.args, shell=self.shell, cwd=self.cwd, **kwargs)


@dataclass(frozen=True)
class ClusterJob:
    run_id: str
    config_path: str
    script_path: str
    log_path: str
    submit: SchedulerCommand


def prepare_cluster_job(config, settings, run_id) -> ClusterJob:
    """Write <output_dir>/prism_job_<id>.json and prism_run_<id>.sh; return their paths."""
    out = config.output_dir
    config_path = os.path.join(out, f"prism_job_{run_id}.json")
    script_path = os.path.join(out, f"prism_run_{run_id}.sh")
    log_path = os.path.join(out, f"prism_log_{run_id}.txt")

    with open(config_path, "w") as f:
        f.write(config.to_json())
    # newline="\n": this is a bash script even when the GUI runs on Windows.
    with open(script_path, "w", newline="\n") as f:
        f.write(render_job_script(settings, config_path, log_path))

    return ClusterJob(run_id, config_path, script_path, log_path,
                      submit_command(settings, out, script_path))


def render_job_script(settings, config_path, log_path) -> str:
    log = shlex.quote(log_path)
    return f"""#!/bin/bash
{settings['headers']}

{settings['env_setup']}

# Run this checkout's headless worker (it imports PRISM from the folder above it, so the node runs the
# same version as the GUI). -u: the GUI tails this log, so output must not sit in a buffer.
python -u {shlex.quote(HEADLESS_SCRIPT)} --config {shlex.quote(config_path)} > {log} 2>&1

# If the python script or module loads fail, bash will register a non-zero exit code.
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "[CRITICAL ERROR] Remote job died with system exit code $EXIT_CODE." >> {log}
fi
"""


def submit_command(settings, output_dir, script_path) -> SchedulerCommand:
    return scheduler_command(settings, f"{settings['submit_cmd']} {shlex.quote(os.path.basename(script_path))}",
                             cwd=output_dir)


def scheduler_command(settings, text, cwd=None) -> SchedulerCommand:
    """`text`, a shell command as the user wrote it, run in `cwd`: through the SSH prefix if one is
    set (the cluster's login shell reads it), otherwise by the local shell."""
    prefix = settings.get("ssh_prefix", "").strip()
    if not prefix:
        return SchedulerCommand(text, shell=True, cwd=cwd)
    remote = f"cd {shlex.quote(cwd)} && {text}" if cwd else text
    return SchedulerCommand(split_command(prefix) + [remote], shell=False)


def split_command(text):
    """A command line typed in Cluster Settings (e.g. the SSH prefix) as program and arguments.
    On Windows, backslashes in paths are kept and "double quotes" group words."""
    if os.name != "nt":
        return shlex.split(text)
    words = shlex.split(text, posix=False)
    return [w[1:-1] if len(w) >= 2 and w[0] == w[-1] == '"' else w for w in words]


def parse_job_id(submit_output) -> str | None:
    """The job ID from the submit command's output: its last word (SLURM: "Submitted batch job
    4242"; PBS: "4242.server"), or None if that doesn't look like a job ID."""
    words = (submit_output or "").split()
    return words[-1] if words and JOB_ID.fullmatch(words[-1]) else None


def job_state(settings, job_id, run=subprocess.run):
    """True while the scheduler still lists the job, False once it does not, None when that
    cannot be told (no status command, no job ID, or the command itself failed)."""
    template = settings.get("status_cmd", "").strip()
    if not template or not job_id or not JOB_ID.fullmatch(job_id):
        return None
    try:
        res = scheduler_command(settings, template.format(job_id=job_id)).run(
            run, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if (res.stdout or "").strip():
        return True
    # squeue prints nothing for a finished job, and "Invalid job id" once it has been purged.
    if res.returncode == 0 or "invalid job id" in (res.stderr or "").lower():
        return False
    return None


def cancel_job(settings, job_id, run=subprocess.run) -> str:
    """Ask the scheduler to cancel the job; returns a line for the log."""
    template = settings.get("cancel_cmd", "").strip()
    if not template:
        return "[WARNING] No cancel command is set, so the job keeps running on the cluster."
    if not job_id or not JOB_ID.fullmatch(job_id):
        return "[WARNING] PRISM has no job ID to cancel, so the job keeps running on the cluster."
    try:
        res = scheduler_command(settings, template.format(job_id=job_id)).run(
            run, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        return f"[WARNING] Could not cancel job {job_id}: {e}"
    if res.returncode != 0:
        return f"[WARNING] Could not cancel job {job_id}: {(res.stderr or res.stdout).strip()}"
    return f"[CLUSTER] Cancelled job {job_id}."


def watch_job(log_path, job_id, settings, *, log, progress, status, sleep, run=subprocess.run,
              clock=time.monotonic):
    """
    Follow a submitted job until its log says how it ended, relaying log lines to `log` and
    [PROGRESS] values to `progress`, and the job's phase to `status`.

    Returns "complete" ([SUCCESS] in the log), "failed" ([CRITICAL ERROR]) or "lost": the
    scheduler no longer lists the job, yet after STATUS_GRACE_SECONDS its log still has no
    verdict, e.g. because the scheduler killed it at its time or memory limit.

    The log is read incrementally (only new bytes, a partly written last line held back).
    After a verdict, reading continues until the log has been quiet for two polls, so the
    traceback and exit-code lines written just after it are shown too. `sleep(seconds)` waits
    between polls; raising core.cancel.Cancelled from it stops the watch.
    """
    started = clock()
    next_check = started + STATUS_POLL_SECONDS
    gone_since, warned, verdict, quiet = None, False, None, 0
    pending, handle = "", None
    try:
        while True:
            if handle is None and os.path.exists(log_path):
                handle = open(log_path, "r", errors="replace")
                status("Executing on compute node...")
            new = handle.read() if handle is not None else ""
            if new:
                quiet = 0
                lines = (pending + new).split("\n")
                pending = lines.pop()
                for line in lines:
                    verdict = _relay(line, log, progress) or verdict
                continue
            if verdict is not None:
                quiet += 1
                if quiet >= 2:
                    if pending:
                        _relay(pending, log, progress)
                    return verdict
            elif handle is None:
                status(f"Waiting in queue... ({_elapsed(clock() - started)})")

            now = clock()
            if verdict is None and now >= next_check:
                next_check = now + STATUS_POLL_SECONDS
                state = job_state(settings, job_id, run)
                if state is None and not warned and job_id and settings.get("status_cmd", "").strip():
                    warned = True
                    log(f"[WARNING] Could not ask the scheduler about job {job_id}, so PRISM cannot tell if it "
                        "is killed. Cancel stops watching.")
                if state is False and gone_since is None:
                    gone_since = now
                elif state is True:
                    gone_since = None
            if verdict is None and gone_since is not None and now - gone_since >= STATUS_GRACE_SECONDS:
                if pending:
                    _relay(pending, log, progress)
                log(f"[ERROR] The scheduler no longer lists job {job_id}, but its log never said whether it "
                    "finished. It was probably stopped by the scheduler (time or memory limit) or cancelled; "
                    "the scheduler's own output for the job (for SLURM, slurm-<job id>.out) says why.")
                return "lost"
            sleep(LOG_POLL_SECONDS)
    finally:
        if handle is not None:
            handle.close()


def _relay(line, log, progress):
    """Pass one log line on; returns "complete"/"failed" if it carries the job's verdict."""
    line = line.strip()
    if not line:
        return None
    if "[PROGRESS]" in line:
        try:
            progress(float(line.split("[PROGRESS]")[1].strip()))
        except ValueError:
            pass
        return None
    log(line)
    if "[SUCCESS]" in line:
        return "complete"
    if "[CRITICAL ERROR]" in line:
        return "failed"
    return None


def _elapsed(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes} min {seconds:02d} s" if minutes else f"{seconds} s"
