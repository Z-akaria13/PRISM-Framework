"""
EXECUTION SCREEN
Provides the UI to configure batch size, output directory, and trigger
the Extraction engine (either Locally or via a Remote Cluster Dispatcher).
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import dataclasses
import os
import uuid
import subprocess

from prism.core.cache import AuditCache, CacheError, FAILED_NAME, results_exist
from prism.core.cancel import Cancelled
from prism.core.config import DEVICE_CHOICES, AuditConfig
from prism.core.dispatch import DEFAULT_CLUSTER_SETTINGS, cancel_job, parse_job_id, prepare_cluster_job, watch_job
from prism.core.execution import run_extraction
from prism.core.subject import Subject
from prism.gui.utils import create_info_button
from prism.gui.worker import Task

class ExecutionFrame(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app

        # Default Cluster Profile
        self.cluster_config = dict(DEFAULT_CLUSTER_SETTINGS)
        self.task = None               # the running extraction or cluster watch
        self._cluster_job = None       # its job ID once submitted
        self._kill_cluster_job = False  # on Cancel: also cancel the job on the cluster

        self._init_ui()

    def _init_ui(self):
        settings_frame = ttk.LabelFrame(self, text="Execution Settings", padding=20)
        settings_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Label(settings_frame, text="Batch Size:").grid(row=0, column=0, sticky="w", pady=5)
        create_info_button(settings_frame, "Batch Size", "How many items to run at once. Reduce to 4 or 8 if you encounter Out-Of-Memory (OOM) crashes.").grid(row=0, column=1, padx=(0,5))
        self.ent_batch = ttk.Entry(settings_frame, width=10)
        self.ent_batch.insert(0, "16")
        self.ent_batch.grid(row=0, column=2, sticky="w", padx=5)
        
        ttk.Label(settings_frame, text="Output Directory:").grid(row=1, column=0, sticky="w", pady=5)
        create_info_button(settings_frame, "Output Directory", "Where to save the massive HDF5 embedding file. Must be accessible by the remote cluster if dispatching.").grid(row=1, column=1, padx=(0,5))
        self.ent_outdir = ttk.Entry(settings_frame, width=50)
        self.ent_outdir.grid(row=1, column=2, padx=5)
        ttk.Button(settings_frame, text="Browse", command=self.browse_outdir).grid(row=1, column=3)

        ttk.Label(settings_frame, text="Device:").grid(row=2, column=0, sticky="w", pady=5)
        create_info_button(settings_frame, "Device", "Where the model runs during extraction. 'auto' uses a GPU if the machine running the job has one (for cluster jobs, that is the compute node).").grid(row=2, column=1, padx=(0,5))
        self.cb_device = ttk.Combobox(settings_frame, values=DEVICE_CHOICES, state="readonly", width=8)
        self.cb_device.set("auto")
        self.cb_device.grid(row=2, column=2, sticky="w", padx=5)

        ttk.Label(settings_frame, text="Stop if more than:").grid(row=3, column=0, sticky="w", pady=5)
        create_info_button(settings_frame, "Failed Rows",
            "If more than this share of rows fails to load or run, the run stops and saves nothing. "
            f"Below it, failed rows are left out and listed in {FAILED_NAME}. 100 never stops."
        ).grid(row=3, column=1, padx=(0,5))
        limit_row = ttk.Frame(settings_frame)
        limit_row.grid(row=3, column=2, sticky="w", padx=5)
        self.ent_max_failed = ttk.Entry(limit_row, width=6)
        self.ent_max_failed.insert(0, f"{AuditConfig.max_failed_percent:g}")
        self.ent_max_failed.pack(side=tk.LEFT)
        ttk.Label(limit_row, text="% of rows fail").pack(side=tk.LEFT, padx=(4, 0))

        # --- Run Mode & Cluster Config ---
        run_frame = ttk.Frame(self)
        run_frame.pack(fill=tk.X, padx=20, pady=5)
        
        self.run_mode = tk.StringVar(value="local")
        ttk.Radiobutton(run_frame, text="Run Locally", variable=self.run_mode, value="local").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(run_frame, text="Dispatch to Cluster", variable=self.run_mode, value="cluster").pack(side=tk.LEFT, padx=10)
        
        ttk.Button(run_frame, text="⚙️ Cluster Settings", command=self.open_cluster_settings).pack(side=tk.LEFT, padx=20)

        # --- Run / Cancel ---
        button_row = ttk.Frame(self)
        button_row.pack(pady=15)
        self.btn_run = ttk.Button(button_row, text="START EXTRACTION", command=self.start_execution)
        self.btn_run.pack(side=tk.LEFT, padx=5)
        self.btn_cancel = ttk.Button(button_row, text="Cancel", command=self.cancel_execution, state="disabled")
        self.btn_cancel.pack(side=tk.LEFT, padx=5)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=40, pady=5)
        
        self.lbl_status = ttk.Label(self, text="Ready.")
        self.lbl_status.pack()

        self.txt_console = tk.Text(self, height=12, state="disabled", bg="#1e1e1e", fg="#00ff00")
        self.txt_console.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

    def open_cluster_settings(self):
        win = tk.Toplevel(self)
        win.title("Cluster Configuration")
        win.geometry("560x640")
        
        ttk.Label(win, text="SSH Tunnel Prefix (Leave blank if running on Head Node):").pack(anchor="w", padx=10, pady=(10,0))
        ent_ssh = ttk.Entry(win, width=60)
        ent_ssh.insert(0, self.cluster_config["ssh_prefix"])
        ent_ssh.pack(padx=10, pady=2)
        
        ttk.Label(win, text="Submission Command (e.g., sbatch, qsub):").pack(anchor="w", padx=10, pady=(10,0))
        ent_cmd = ttk.Entry(win, width=20)
        ent_cmd.insert(0, self.cluster_config["submit_cmd"])
        ent_cmd.pack(anchor="w", padx=10, pady=2)

        ttk.Label(win, text="Job Headers (Resource Requests):").pack(anchor="w", padx=10, pady=(10,0))
        txt_head = tk.Text(win, height=5, width=60)
        txt_head.insert("1.0", self.cluster_config["headers"])
        txt_head.pack(padx=10, pady=2)

        ttk.Label(win, text="Environment Setup (Modules/Conda):").pack(anchor="w", padx=10, pady=(10,0))
        txt_env = tk.Text(win, height=3, width=60)
        txt_env.insert("1.0", self.cluster_config["env_setup"])
        txt_env.pack(padx=10, pady=2)

        ttk.Label(win, text="Status Command ({job_id}; prints nothing once the job has left the queue; blank: never ask):").pack(anchor="w", padx=10, pady=(10,0))
        ent_status = ttk.Entry(win, width=40)
        ent_status.insert(0, self.cluster_config.get("status_cmd", ""))
        ent_status.pack(anchor="w", padx=10, pady=2)

        ttk.Label(win, text="Cancel Command ({job_id}):").pack(anchor="w", padx=10, pady=(10,0))
        ent_cancel = ttk.Entry(win, width=40)
        ent_cancel.insert(0, self.cluster_config.get("cancel_cmd", ""))
        ent_cancel.pack(anchor="w", padx=10, pady=2)

        def save():
            self.cluster_config["ssh_prefix"] = ent_ssh.get()
            self.cluster_config["submit_cmd"] = ent_cmd.get()
            self.cluster_config["headers"] = txt_head.get("1.0", "end-1c")
            self.cluster_config["env_setup"] = txt_env.get("1.0", "end-1c")
            self.cluster_config["status_cmd"] = ent_status.get()
            self.cluster_config["cancel_cmd"] = ent_cancel.get()
            win.destroy()

        ttk.Button(win, text="Save Settings", command=save).pack(pady=20)

    def browse_outdir(self):
        path = filedialog.askdirectory()
        if path:
            self.ent_outdir.delete(0, tk.END)
            self.ent_outdir.insert(0, path)

    def log(self, msg):
        self.txt_console.config(state="normal")
        self.txt_console.insert(tk.END, f"{msg}\n")
        self.txt_console.see(tk.END)
        self.txt_console.config(state="disabled")

    def update_progress(self, val):
        self.progress_var.set(val)

    def build_config(self, output_dir, batch_size):
        """The job described by the tabs: Setup (subject), Evidence (columns), Wiretap (layers), this tab."""
        app = self.app
        return AuditConfig(
            project_root=os.path.abspath(app.project_root or ""),
            script_path=os.path.abspath(app.script_path),
            framework=app.framework,
            csv_path=os.path.abspath(app.csv_path),
            id_col=app.config_id_col,
            target_col=app.config_target_col,
            group_col=app.config_group_col,
            layers=app.frame_wiretap.extracted_layers(),
            batch_size=batch_size,
            output_dir=os.path.abspath(output_dir),
            device=self.cb_device.get(),
            max_failed_percent=_percentage(self.ent_max_failed.get()),
        )

    def start_execution(self):
        if not self.app.model_subject:
            messagebox.showerror("Error", "No Subject Loaded. Go to Setup tab.")
            return
        if self.app.metadata_df is None:
            messagebox.showerror("Error", "No Evidence Loaded. Go to Evidence tab.")
            return
        if not self.ent_outdir.get():
            messagebox.showerror("Error", "Please select an output directory.")
            return
        shapes = self.app.frame_wiretap.task
        if shapes is not None and shapes.running:
            messagebox.showinfo("Busy", "Wait for Show shapes in the Wiretap tab to finish.")
            return
        try:
            config = self.build_config(self.ent_outdir.get(), int(self.ent_batch.get()))
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid execution settings:\n{e}")
            return
        if results_exist(config.output_dir):
            if not messagebox.askyesno("Replace existing results?",
                                       f"{config.output_dir} already holds extraction results.\n\n"
                                       "Replace them? They stay until the new run finishes. "
                                       "Choose No to pick another output directory."):
                return
            config = dataclasses.replace(config, overwrite=True)

        self.btn_run.config(state="disabled")
        self.btn_cancel.config(state="normal")
        self.progress_var.set(0)

        if self.run_mode.get() == "local":
            self.lbl_status.config(text="Extracting features locally...")
            self.log("Initializing Execution Engine...")
            subject = Subject(self.app.connector_instance, self.app.model_subject, self.app.inspector)
            self.run_local(config, subject)
        else:
            self.lbl_status.config(text="Dispatching to Cluster...")
            self.run_cluster(config)

    def cancel_execution(self):
        """Stop the running extraction, or stop watching the cluster job (and, if wanted, cancel it)."""
        task = self.task
        if task is None or not task.running:
            return
        if self._cluster_job and self.cluster_config.get("cancel_cmd", "").strip():
            self._kill_cluster_job = messagebox.askyesno(
                "Cancel the cluster job too?",
                f"Stop watching job {self._cluster_job}.\n\nAlso cancel the job on the cluster? Choose No to leave it "
                "running: its results will still be written to the output directory.")
        self.lbl_status.config(text="Cancelling...")
        self.btn_cancel.config(state="disabled")
        task.cancel()

    def run_local(self, config, subject):
        """Run the extraction here, off the Tk thread; returns the Task."""
        metadata = self.app.metadata_df

        def work(task):
            return run_extraction(config, subject, metadata, task.progress, task.log, check=task.check)

        def done(success):
            if success:
                self.app.active_cache_dir = config.output_dir
                self._show_outcome(config.output_dir, "Extraction Complete!")
            else:
                self.lbl_status.config(text="Extraction Failed. Check console.")

        return self._start(work, done, "Extraction cancelled; nothing was saved.")

    def run_cluster(self, config):
        """Submit the job, then follow its log and the scheduler's queue off the Tk thread; returns the Task."""
        settings = dict(self.cluster_config)
        hidden = getattr(self.app.inspector, 'extracts_hidden_layers', True)
        self._cluster_job, self._kill_cluster_job = None, False

        def work(task):
            if not config.layers and hidden:
                task.log("[ERROR] No layers chosen: tick the layers to extract in the Wiretap tab.")
                return "not submitted"
            job = prepare_cluster_job(config, settings, str(uuid.uuid4())[:6])
            task.log(f"Submitting job via {settings['submit_cmd']}...")
            res = job.submit.run(subprocess.run, capture_output=True, text=True)
            if res.returncode != 0:
                task.log(f"[ERROR] Submit failed: {res.stderr}")
                return "not submitted"
            task.log(f"[CLUSTER] Job Accepted: {res.stdout.strip()}")
            job_id = self._cluster_job = parse_job_id(res.stdout)
            if job_id is None:
                task.log("[WARNING] PRISM could not read a job ID from that output, so it cannot ask the scheduler "
                         "about the job or cancel it. It still follows the job's log.")
            try:
                return watch_job(job.log_path, job_id, settings, log=task.log, progress=task.progress,
                                 status=task.status, sleep=task.sleep, run=subprocess.run)
            except Cancelled:
                if self._kill_cluster_job:
                    task.log(cancel_job(settings, job_id, run=subprocess.run))
                else:
                    task.log(f"[CLUSTER] Stopped watching job {job_id}. It keeps running; its results will "
                             f"appear in {config.output_dir}.")
                raise

        def done(outcome):
            if outcome == "complete":
                self.log("[SUCCESS] Remote Extraction Complete.")
                self.app.active_cache_dir = config.output_dir
                self._show_outcome(config.output_dir, "Complete!",
                                   hint="The output directory must be on a filesystem this computer shares "
                                        "with the cluster.")
            elif outcome == "failed":
                self.log("[ERROR] Remote extraction failed. See the log above.")
                self.lbl_status.config(text="Extraction Failed. Check console.")
            elif outcome == "lost":
                self.lbl_status.config(text="The job ended without finishing. Check console.")
            else:
                self.lbl_status.config(text="Not submitted. Check console.")

        return self._start(work, done, "Stopped watching the cluster job.")

    def _show_outcome(self, output_dir, complete_text, hint=""):
        """Say the run finished, and warn if it left rows out: every analysis will be missing them."""
        try:
            cache = AuditCache.open(output_dir)
        except CacheError as e:
            self.log(f"[ERROR] The results can't be opened: {e}" + (f"\n{hint}" if hint else ""))
            self.lbl_status.config(text="Extraction finished, but its results can't be opened. Check console.")
            return
        if not cache.failure_note:
            self.lbl_status.config(text=complete_text)
            return
        self.lbl_status.config(text=f"Extraction complete: {cache.rows_failed} of {cache.rows_attempted} rows "
                                    "were left out. Check console.")
        messagebox.showwarning("Some rows were left out", cache.failure_note)

    def _start(self, work, done, cancelled_text):
        def failed(error, trace):
            self.log(f"[CRASH] {error}")
            self.log(trace)
            self.lbl_status.config(text="Extraction Failed. Check console.")

        def finish():
            self.btn_run.config(state="normal")
            self.btn_cancel.config(state="disabled")

        # Looked up on every call, so a wrapped log or progress method (as tests do) is honoured.
        self.task = Task(self, work, on_done=done, on_error=failed, on_finish=finish,
                    on_cancel=lambda: self.lbl_status.config(text=cancelled_text),
                    on_progress=lambda value: self.update_progress(value),
                    on_log=lambda message: self.log(message),
                    on_status=lambda status: self.lbl_status.config(text=status)).start()
        return self.task


def _percentage(text):
    """The failed-rows limit as typed; AuditConfig checks its range."""
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"'Stop if more than' must be a percentage from 0 to 100, not '{text}'.") from None
