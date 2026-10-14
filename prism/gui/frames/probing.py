"""
PROBING SCREEN (White Box)
Trains Linear Probes on hidden layer embeddings to detect "Clever Hans" shortcuts.
"""
import tkinter as tk
from tkinter import ttk, messagebox

from prism.analysis import DEFAULT_CONTROLS, MAX_CONTROLS, PROBE_MODELS, linear_probe
from prism.gui.utils import column_mapping, create_info_button, open_cache
from prism.gui.worker import Task


class ProbingFrame(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.cache = None
        self.task = None
        self._init_ui()

    def _init_ui(self):
        # Controls
        ctrl_frame = ttk.LabelFrame(self, text="Linear Probe Config", padding=10)
        ctrl_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(ctrl_frame, text="Load Cache", command=self.load_options).grid(row=0, column=0, padx=5)

        ttk.Label(ctrl_frame, text="Source Layer:").grid(row=0, column=1, padx=(15, 2))
        self.cb_layer = ttk.Combobox(ctrl_frame, state="readonly", width=15)
        self.cb_layer.grid(row=0, column=2, padx=2)

        ttk.Label(ctrl_frame, text="Target Concept:").grid(row=0, column=3, padx=(15, 2))
        create_info_button(ctrl_frame, "Target Concept", "The metadata feature you suspect the model is using as a shortcut (e.g. 'Scanner_Type').").grid(row=0, column=4, padx=(0,5))
        self.cb_concept = ttk.Combobox(ctrl_frame, state="readonly", width=15)
        self.cb_concept.grid(row=0, column=5, padx=2)

        self.btn_run = ttk.Button(ctrl_frame, text="RUN PROBE", command=self.run_probe)
        self.btn_run.grid(row=0, column=6, padx=(20, 5))
        self.btn_cancel = ttk.Button(ctrl_frame, text="Cancel", command=self.cancel_probe, state="disabled")
        self.btn_cancel.grid(row=0, column=7, padx=5)

        self.var_control = tk.BooleanVar(value=True)
        chk = ttk.Checkbutton(ctrl_frame, text="Control Probe", variable=self.var_control)
        chk.grid(row=1, column=0, sticky="w", padx=5, pady=(5,0))
        create_info_button(ctrl_frame, "Control Probe",
            "Re-runs the probe on shuffled labels (whole subjects' labels when Group By is set) to see what score pure chance gets on this data, and reports how many shuffles matched or beat the real score.\n\n"
            "p = (shuffles that matched + 1) / (shuffles + 1), so the lowest p you can see is 1 / (shuffles + 1): 50 shuffles can show p = 0.020, 999 can show p = 0.001.\n\n"
            f"Choose 1 to {MAX_CONTROLS} shuffles. The streamed models (Shrinkage LDA, Ridge) handle hundreds cheaply; Multinomial Logistic refits for every shuffle, so many shuffles are slow.").grid(row=1, column=1, sticky="w", pady=(5,0))
        ttk.Label(ctrl_frame, text="Shuffles:").grid(row=1, column=2, sticky="e", padx=(10, 2), pady=(5,0))
        self.var_shuffles = tk.StringVar(value=str(DEFAULT_CONTROLS))
        ttk.Spinbox(ctrl_frame, from_=1, to=MAX_CONTROLS, increment=1, width=6,
                    textvariable=self.var_shuffles).grid(row=1, column=3, sticky="w", pady=(5,0))
        self.lbl_min_p = ttk.Label(ctrl_frame, text="")
        self.lbl_min_p.grid(row=1, column=4, columnspan=2, sticky="w", padx=5, pady=(5,0))
        self.var_shuffles.trace_add("write", lambda *_: self._show_min_p())
        self._show_min_p()

        ttk.Label(ctrl_frame, text="Probe Model:").grid(row=2, column=0, sticky="w", padx=5, pady=(5,0))
        self.var_model = tk.StringVar(value=PROBE_MODELS[0])
        for column, (value, label) in enumerate([("lda", "Shrinkage LDA"), ("logistic", "Multinomial Logistic"),
                                                 ("ridge", "Ridge Classifier")], start=1):
            ttk.Radiobutton(ctrl_frame, text=label, variable=self.var_model, value=value).grid(row=2, column=column, sticky="w", pady=(5,0))
        create_info_button(ctrl_frame, "Probe Model",
            "All three are linear probes asking the same question: can the concept be read off this layer with a straight cut?\n\n"
            "Shrinkage LDA (recommended, default): fast and exact, streamed from disk without loading the layer; shuffled-label controls are cheap (each fold is factorised once, then every shuffle is a small update). The strongest linear default for several classes, including ordered ones, without masking.\n\n"
            "Multinomial Logistic: the standard baseline and the gold standard for probabilities. Loads the whole layer into memory, and each control is a full refit, so it is the slowest.\n\n"
            "Ridge Classifier: fast and exact, streamed like LDA, with equally cheap controls. Fine for two classes; with three or more, a class lying between others (e.g. a middle severity level) can be masked.\n\n"
            "The streamed models keep running totals of size features x features, so they handle layers up to about 10,000 features per row (any number of rows); wider ones, such as flattened 3D maps, need pooling or the logistic model. Numeric concepts (regression) always use streamed ridge regression.").grid(row=2, column=4, sticky="w", pady=(5,0))
        self.progress_var = tk.DoubleVar()
        ttk.Progressbar(self, variable=self.progress_var, maximum=1.0).pack(fill=tk.X, padx=10)
        self.txt_res = tk.Text(self, height=15, bg="#1e1e1e", fg="#00ff00", font=("Courier", 11))
        self.txt_res.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def _shuffle_count(self):
        """The Shuffles box as a number from 1 to MAX_CONTROLS, or None if it isn't one."""
        try:
            n = int(self.var_shuffles.get().strip())
        except ValueError:
            return None
        return n if 1 <= n <= MAX_CONTROLS else None

    def _show_min_p(self):
        n = self._shuffle_count()
        self.lbl_min_p.config(text=f"lowest possible p: {1 / (n + 1):.3f}" if n else f"enter 1 to {MAX_CONTROLS}")

    def load_options(self):
        cache = open_cache(self.app)
        if cache is None: return
        self.cache = cache
        self.df = cache.metadata
        self.cb_concept['values'] = list(self.df.columns)

        layers = cache.layer_names
        self.cb_layer['values'] = layers

        if layers: self.cb_layer.current(0)
        if len(self.df.columns) > 0: self.cb_concept.current(0)

        self.log("Cache loaded. Select a layer and a concept to probe.")
        for note in cache.warnings:
            self.log(f"[NOTE] {note}")
        _, _, group_col = column_mapping(self.app, cache)
        if group_col:
            self.log(f"Cross-validation folds keep each '{group_col}' group together.")

    def log(self, msg):
        self.txt_res.insert(tk.END, f"{msg}\n")
        self.txt_res.see(tk.END)

    def run_probe(self):
        layer = self.cb_layer.get()
        concept = self.cb_concept.get()
        if self.task is not None and self.task.running: return
        if self.cache is None:
            messagebox.showinfo("No Results", "Load results first.")
            return
        if not self.cache.layer_names:
            messagebox.showinfo("No Layers", "These results have no layers to probe.")
            return
        if not layer or not concept:
            messagebox.showinfo("Nothing Chosen", "Choose a layer and a concept.")
            return

        self.log(f"\n--- PROBING: Can '{layer}' predict '{concept}'? ---")

        model = self.var_model.get()
        if model == "logistic":
            # The logistic probe loads the layer into memory: warn first if that is a lot
            memory_gb = self.cache.layer_nbytes(layer) / (1024**3)
            if memory_gb > 4.0: # If larger than 4GB
                messagebox.showwarning("Memory Warning",
                    f"This layer requires {memory_gb:.1f} GB of RAM. Loading this might freeze your computer. "
                    "Consider a streamed probe model (Shrinkage LDA or Ridge), or re-extracting with 'Mean Pool' instead of 'Flatten'.")

        _, _, group_col = column_mapping(self.app, self.cache)
        groups = self.df[group_col] if group_col and group_col in self.df.columns else None
        if groups is None:
            self.log("[NOTE] No Group By column is set, so folds are split row by row. If subjects have "
                     "several rows (e.g. repeat scans), the probe can score by recognising them; "
                     "set Group By in the Evidence tab.")

        n_controls = self._shuffle_count() if self.var_control.get() else 0
        if n_controls is None:
            messagebox.showerror("Invalid shuffles", f"Shuffles must be a whole number from 1 to {MAX_CONTROLS}.")
            return
        extra = f" plus {n_controls} shuffled-label controls" if n_controls else ""
        cache, target = self.cache, self.df[concept]

        def work(task):
            with cache.open_layer(layer) as features:  # stays on disk; the probe reads what it needs
                task.log(f"Data: {features.shape}. Running cross-validation{extra}...")
                return linear_probe(features, target, groups=groups, n_controls=n_controls, model=model,
                                    progress=task.progress)

        def failed(error, trace):
            self.log(f"[CRASH] {error}")
            print(trace)
            messagebox.showerror("Probe failed", str(error))

        def finish():
            self.btn_run.config(state="normal")
            self.btn_cancel.config(state="disabled")

        self.btn_run.config(state="disabled")
        self.btn_cancel.config(state="normal")
        self.progress_var.set(0)
        self.task = Task(self, work, on_done=lambda result: self._report(result, group_col), on_error=failed,
                         on_cancel=lambda: self.log("[CANCELLED] Probe stopped."), on_finish=finish,
                         on_log=self.log, on_progress=self.progress_var.set).start()
        return self.task

    def cancel_probe(self):
        if self.task is not None and self.task.running:
            self.btn_cancel.config(state="disabled")
            self.task.cancel()

    def _report(self, result, group_col):
        if result["task"] == "regression":
            task = "Regression (Ridge)"
        else:
            task = {"lda": "Classification (Shrinkage LDA, streamed)", "ridge": "Classification (Ridge classifier, streamed)",
                    "logistic": "Classification (LogReg)"}[result["model"]]
        self.log(f"Detected Task: {task}")
        if result["grouped"]:
            self.log(f"Folds keep each '{group_col}' group together ({result['n_groups']} groups).")
        for note in result["notes"]:
            self.log(f"[NOTE] {note}")
        if result["error"]:
            self.log(f"[ERROR] {result['error']}")
            messagebox.showwarning("Cannot run probe", result["error"])
            return

        if result["task"] == "regression":
            self.log(f"RESULT (R² Score): {result['r2_mean']:.3f} (± {result['r2_std']:.3f})")
        else:
            self.log(f"RESULT (AUC): {result['auc_mean']:.3f} (± {result['auc_std']:.3f})")
            self.log(f"Balanced accuracy: {result['balanced_accuracy_mean']:.3f} "
                     f"(± {result['balanced_accuracy_std']:.3f}), chance: {result['chance_balanced_accuracy']:.3f}")

        control = result["control"]
        if control:
            self.log(f"Control ({control['n']} label shuffles, per {control['shuffled']}): "
                     f"{result['metric']} {control['mean']:.3f} (± {control['std']:.3f})")
            self.log(f"  {control['n_at_least_real']} of {control['n']} shuffles matched or beat the real probe "
                     f"(p = {control['p_value']:.3f}).")
            if control["p_value"] > 0.05:
                self.log("WARNING: the real probe does not clearly beat shuffled labels, so this is no "
                         "reliable evidence that the layer encodes the concept.")
