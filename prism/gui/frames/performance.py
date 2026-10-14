import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, classification_report
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from prism.analysis import (DEFAULT_BOOTSTRAP, DISPARITY_THRESHOLD, MIN_GROUP_SIZE, REGRESSION_MIN_UNIQUE,
                      fairness_audit, metric_label)
from prism.core.labels import class_order, decode_predictions
from prism.gui.utils import column_mapping, create_info_button, open_cache
from prism.gui.worker import Runner

MAX_GROUP_SIZE = 1000  # upper end of the Min Group Size box


class PerformanceFrame(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.figure = None
        self.canvas = None
        self.cache = None
        self.raw_preds = None
        self.y_raw = None
        self.target_col = None
        self.group_col = None
        self.result = None
        self.task_type = "classification"
        self.unique_classes = []  # class labels in model-output order
        self.notes = []           # caveats shown under every report
        self.runner = Runner(self)  # the bootstrap runs off the Tk thread; a newer request replaces an older one
        self._init_ui()

    def _init_ui(self):
        ctrl_frame = ttk.LabelFrame(self, text="Analysis Controls", padding=10)
        ctrl_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(ctrl_frame, text="Load Cached Results", command=self.load_results).grid(row=0, column=0, padx=5)

        ttk.Label(ctrl_frame, text="Positive Class:").grid(row=0, column=1, padx=(15, 2))
        create_info_button(ctrl_frame, "Positive Class", "Dictates which class is considered 'Positive' (e.g. Disease) when calculating Sensitivity and False Positive Rates.").grid(row=0, column=2, padx=(0,5))
        self.cb_pos_class = ttk.Combobox(ctrl_frame, state="readonly", width=15)
        self.cb_pos_class.grid(row=0, column=3, padx=2)
        self.cb_pos_class.bind("<<ComboboxSelected>>", self.run_analysis)

        ttk.Label(ctrl_frame, text="Slice By:").grid(row=0, column=4, padx=(15, 2))
        create_info_button(ctrl_frame, "Slice By", "Select a demographic feature (e.g. Race, Sex) to check for fairness violations and disparate impact.").grid(row=0, column=5, padx=(0,5))
        self.cb_slice = ttk.Combobox(ctrl_frame, state="readonly", width=20)
        self.cb_slice.grid(row=0, column=6, padx=2)
        self.cb_slice.bind("<<ComboboxSelected>>", self.run_analysis)

        ttk.Label(ctrl_frame, text="Min Group Size:").grid(row=0, column=7, padx=(15, 2))
        create_info_button(ctrl_frame, "Min Group Size",
            "Groups smaller than this are still shown, but are left out of the disparity scores: with a handful of people, a large gap is easily chance.\n\n"
            "Counted in subjects when the Group By column is set (each person once, however many scans), otherwise in scans.\n\n"
            "Set Group By whenever people have more than one scan: scans of one person tend to agree, so treating them as independent makes the intervals too narrow.\n\n"
            f"Every figure comes with a 95% confidence interval from {DEFAULT_BOOTSTRAP} bootstrap resamples (of whole subjects when Group By is set). "
            f"A disparity above {DISPARITY_THRESHOLD} is only flagged when its interval excludes 0; otherwise it is marked as possible noise.").grid(row=0, column=8, padx=(0, 5))
        self.var_min_group = tk.StringVar(value=str(MIN_GROUP_SIZE))
        spin = ttk.Spinbox(ctrl_frame, from_=1, to=MAX_GROUP_SIZE, increment=1, width=5,
                           textvariable=self.var_min_group, command=self.run_analysis)
        spin.grid(row=0, column=9, padx=2)
        spin.bind("<Return>", self.run_analysis)

        self.lbl_stats = ttk.Label(ctrl_frame, text="Global Acc: -- | F1: --", font=("Arial", 10, "bold"))
        self.lbl_stats.grid(row=0, column=10, padx=20)

        content_paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        content_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.plot_frame = ttk.Frame(content_paned)
        content_paned.add(self.plot_frame, weight=3)

        report_frame = ttk.LabelFrame(content_paned, text="Detailed Analytics")
        content_paned.add(report_frame, weight=2)

        self.txt_report = tk.Text(report_frame, bg="#1e1e1e", fg="#00ff00", font=("Courier", 10))
        self.txt_report.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.txt_report.insert(tk.END, "Awaiting data...")
        self.txt_report.config(state="disabled")

    def log_report(self, text, clear=False):
        self.txt_report.config(state="normal")
        if clear:
            self.txt_report.delete(1.0, tk.END)
        self.txt_report.insert(tk.END, f"{text}\n")
        self.txt_report.see(tk.END)
        self.txt_report.config(state="disabled")

    def min_group_size(self):
        """The Min Group Size box as a number from 1 to MAX_GROUP_SIZE (the default if it isn't one)."""
        try:
            n = int(self.var_min_group.get().strip())
        except ValueError:
            return MIN_GROUP_SIZE
        return min(max(n, 1), MAX_GROUP_SIZE)

    def load_results(self):
        cache = open_cache(self.app)
        if cache is None: return
        self.cache = cache
        self.df = cache.metadata
        self.raw_preds = cache.model_output()
        self.notes = list(cache.warnings)

        id_col, target, group_col = column_mapping(self.app, cache)
        if not target:
            target = simpledialog.askstring("Target Column", "What is the name of your Ground Truth column?\n(e.g., 'diagnosis')")
            if not target or target not in self.df.columns:
                messagebox.showerror("Error", f"Column '{target}' not found in CSV.")
                return
            self.app.config_target_col = target
        if target not in self.df.columns:
            messagebox.showerror("Error", f"Column '{target}' not found in CSV.")
            return
        self.target_col = target
        self.group_col = group_col if group_col in self.df.columns else None
        if group_col and self.group_col is None:
            self.notes.append(f"The Group By column '{group_col}' is not in these results, so the intervals "
                              "resample scans, not subjects.")

        self.y_raw = self.df[self.target_col]

        if pd.api.types.is_numeric_dtype(self.y_raw) and self.y_raw.nunique() > REGRESSION_MIN_UNIQUE:
            self.task_type = "regression"
            self.cb_pos_class.set("N/A (Regression)")
            self.cb_pos_class.config(state="disabled")
        else:
            self.task_type = "classification"
            self.cb_pos_class.config(state="readonly")
            self.unique_classes, order_notes = class_order(self.y_raw, cache.class_names)
            self.notes += order_notes
            self.cb_pos_class['values'] = self.unique_classes

            if len(self.unique_classes) > 2:
                self.task_type = "multiclass"

            if "1" in self.unique_classes:
                self.cb_pos_class.set("1")
            else:
                self.cb_pos_class.current(len(self.unique_classes)-1)

        # The ID and Group By columns name people, not demographic groups.
        skip = {self.target_col, id_col, group_col}
        slicer_cols = [col for col in self.df.columns if col not in skip and self.df[col].nunique() <= 20]

        self.cb_slice['values'] = ["(Global)"] + slicer_cols
        self.cb_slice.current(0)

        self.run_analysis()

    def run_analysis(self, event=None):
        """Recompute for the current controls in the background; returns the Task (None if nothing to do)."""
        slice_col = self.cb_slice.get()
        if not slice_col or self.cache is None: return
        audit = dict(slices=None if slice_col == "(Global)" else self.df[slice_col],
                     groups=self.df[self.group_col] if self.group_col else None,
                     min_group_size=self.min_group_size())
        raw, task_type = self.raw_preds, self.task_type

        if task_type == "regression":
            pos_class_str = None
            y_true = self.y_raw.to_numpy(dtype=float)

            def work(task):
                y_pred = np.asarray(raw, dtype=float).flatten()
                return y_true, y_pred, fairness_audit(y_true, y_pred, task="regression", progress=task.progress, **audit)
        else:
            pos_class_str = self.cb_pos_class.get()
            if not pos_class_str: return
            y_true = self.y_raw.astype(str).values
            classes, output_kind = list(self.unique_classes), self.cache.output_kind
            positive = pos_class_str if task_type == "classification" else None

            def work(task):
                y_pred = decode_predictions(raw, classes, output_kind)
                return y_true, y_pred, fairness_audit(y_true, y_pred, classes=classes, positive_class=positive,
                                                      progress=task.progress, **audit)

        self.lbl_stats.config(text="Computing intervals...")
        return self.runner.submit(
            work, on_done=lambda outcome: self._show(slice_col, pos_class_str, *outcome), on_error=self._failed,
            on_progress=lambda done: self.lbl_stats.config(text=f"Computing intervals... {done:.0%}"))

    def _failed(self, error, trace):
        self.lbl_stats.config(text="Analysis failed.")
        messagebox.showerror("Error", str(error))

    def _show(self, slice_col, pos_class_str, y_true, y_pred, result):
        self.y_true, self.y_pred, self.result = y_true, y_pred, result
        g = result["global"]["metrics"]
        if self.task_type == "regression":
            self.lbl_stats.config(text=f"Global R²: {g['r2']['value']:.3f} | RMSE: {g['rmse']['value']:.3f}")
        else:
            g_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            self.lbl_stats.config(text=f"Global Acc: {g['accuracy']['value']*100:.1f}% | Macro F1: {g_f1:.3f}")

        if self.canvas: self.canvas.get_tk_widget().destroy()
        # A plain Figure, not pyplot: pyplot would keep every figure ever drawn alive.
        self.figure = Figure(figsize=(7, 5))
        ax = self.figure.add_subplot()

        if self.task_type == "regression":
            self._render_regression(ax, slice_col, result)
        elif self.task_type == "multiclass":
            if slice_col == "(Global)": self._render_multiclass_global(ax, result)
            else: self._render_multiclass_sliced(ax, slice_col, result)
        elif slice_col == "(Global)":
            self._render_binary_global(ax, pos_class_str, result)
        else:
            self._render_binary_sliced(ax, slice_col, pos_class_str, result)

        self.figure.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        for note in self.notes + result["notes"]:
            self.log_report(f"\n[NOTE] {note}")

    # ------------------------------------------------------------------
    # Shared report pieces
    # ------------------------------------------------------------------

    def _log_intervals(self, result, sliced):
        b = result["bootstrap"]
        if b["replicates"]:
            units = (f"{b['units']} subjects (Group By: {self.group_col})" if b["unit"] == "group"
                     else f"{b['units']} scans")
            self.log_report(f"Intervals     : {b['confidence']:.0%} bootstrap, {b['replicates']} resamples of {units}")
        if sliced:
            unit = "subjects" if b["unit"] == "group" else "scans"
            self.log_report(f"Min group size: {result['min_group_size']} {unit} to enter the disparity scores")
        self.log_report("")

    def _group_header(self, name, block, result):
        header = f"[{name}] (n={block['n']})"
        if result["bootstrap"]["unit"] == "group":
            header += f" from {block['units']} subjects"
        if not block["eligible"]:
            header += f"  [below {result['min_group_size']}: not in the disparity scores]"
        return header

    def _log_gap(self, gap, label, short, warning, signed=False):
        if gap["high"] is None:
            unit = "subjects" if self.result["bootstrap"]["unit"] == "group" else "scans"
            self.log_report(f"{short}: N/A (fewer than two groups of at least {self.result['min_group_size']} "
                            f"{unit} with a defined {metric_label(gap['metric'])})")
            return
        self.log_report(f"{label} : {_with_ci(gap, signed)}  ({gap['high']} vs {gap['low']})")
        if gap["flagged"]:
            self.log_report(f"  [WARNING] {warning}")
        elif gap["noise"]:
            self.log_report(f"  [NOTE] Above {DISPARITY_THRESHOLD}, but the interval includes 0: this gap may be noise.")

    # ------------------------------------------------------------------
    # Binary
    # ------------------------------------------------------------------

    def _render_binary_global(self, ax, pos_class, result):
        g = result["global"]
        m = g["metrics"]
        tn, fp, fn, tp = _binary_counts(g["confusion"], result)
        cm = np.array([[tn, fp], [fn, tp]])
        ax.matshow(cm, cmap='Blues', alpha=0.7)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', size='xx-large')

        ax.set_title(f"Global Confusion Matrix (Pos='{pos_class}')", pad=20)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Negative', 'Positive'])
        ax.set_yticklabels(['Negative', 'Positive'])

        self.log_report("--- GLOBAL PERFORMANCE REPORT ---", clear=True)
        self.log_report(f"Total Samples : {g['n']}")
        self.log_report(f"Positive Class: '{pos_class}'")
        self._log_intervals(result, sliced=False)

        self.log_report("--- METRICS ---")
        self.log_report(f"Accuracy    : {_with_ci(m['accuracy'])}")
        self.log_report(f"Sensitivity : {_with_ci(m['tpr'])} (True Pos Rate)")
        self.log_report(f"Specificity : {_with_ci(m['tnr'])} (True Neg Rate)")
        self.log_report(f"FPR (Alarm) : {_with_ci(m['fpr'])} (False Pos Rate)")
        self.log_report(f"FNR (Miss)  : {_with_ci(m['fnr'])} (False Neg Rate)")
        self.log_report(f"Precision   : {_with_ci(m['precision'])} (Pos Pred Value)")
        self.log_report(f"F1 Score    : {_with_ci(m['f1'])}\n")

        self.log_report("--- RAW COUNTS ---")
        self.log_report(f"True Positives  (Hit)   : {tp}")
        self.log_report(f"True Negatives  (Corr)  : {tn}")
        self.log_report(f"False Positives (Alarm) : {fp}")
        self.log_report(f"False Negatives (Miss)  : {fn}")

    def _render_binary_sliced(self, ax, slice_col, pos_class, result):
        self.log_report(f"--- FAIRNESS REPORT: {slice_col.upper()} ---", clear=True)
        self.log_report(f"Positive Class: '{pos_class}'")
        self._log_intervals(result, sliced=True)

        for name, block in result["groups"].items():
            m = block["metrics"]
            tn, fp, fn, tp = _binary_counts(block["confusion"], result)
            self.log_report(self._group_header(name, block, result))
            self.log_report(f"  Accuracy  : {_with_ci(m['accuracy'])}")
            self.log_report(f"  TPR (Sens): {_with_ci(m['tpr'])} | TNR (Spec): {_with_ci(m['tnr'])}")
            self.log_report(f"  Error Rate: FPR={_with_ci(m['fpr'])} (Alarm) | FNR={_with_ci(m['fnr'])} (Miss)")
            self.log_report(f"  Selection : {_with_ci(m['selection_rate'])} (Flagged Positive)")
            self.log_report(f"  Confusion : TP={tp}, FP={fp}, TN={tn}, FN={fn}\n")

        _grouped_bars(ax, result, [("accuracy", "Accuracy", "lightgray"),
                                   ("tpr", "True Positive Rate (Recall)", "#2ca02c"),
                                   ("selection_rate", "Positive Prediction Rate", "#ff7f0e")])
        ax.set_ylabel('Score (95% CI)')
        ax.set_title(f"Fairness Audit by {slice_col}")
        ax.set_ylim(0, 1.1)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=3)
        global_tpr = result["global"]["metrics"]["tpr"]["value"]
        if np.isfinite(global_tpr):
            ax.axhline(global_tpr, color='green', linestyle=':', linewidth=2, alpha=0.7)

        gaps = result["gaps"]
        self.log_report("--- DISPARITY SCORES ---")
        self._log_gap(gaps["equal_opportunity"], "Equal Opportunity Diff (TPR)", "Equal Opportunity Diff",
                      "High disparity in True Positive Rates.")
        self._log_gap(gaps["predictive_equality"], "Predictive Equality Diff (FPR)", "Predictive Equality Diff",
                      "High disparity in False Positive Rates (Unfair Alarms).")
        self._log_gap(gaps["demographic_parity"], "Demographic Parity Diff (Sel)", "Demographic Parity Diff",
                      "High disparity in Selection Rates.")
        self._log_gap(gaps["accuracy"], "Accuracy Diff", "Accuracy Diff", "High disparity in Accuracy.")

    # ------------------------------------------------------------------
    # Regression
    # ------------------------------------------------------------------

    def _render_regression(self, ax, slice_col, result):
        self.log_report(f"--- REGRESSION REPORT: {slice_col.upper()} ---", clear=True)

        if slice_col == "(Global)":
            ax.scatter(self.y_true, self.y_pred, alpha=0.5, color='blue')
            min_val = min(self.y_true.min(), self.y_pred.min())
            max_val = max(self.y_true.max(), self.y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
            ax.set_title("Predicted vs Actual")
            ax.set_xlabel("Actual Ground Truth")
            ax.set_ylabel("Model Prediction")
            ax.legend()

            m = result["global"]["metrics"]
            self.log_report(f"Total Samples : {result['global']['n']}")
            self._log_intervals(result, sliced=False)
            self.log_report(f"R² Score      : {_with_ci(m['r2'], digits=4)}")
            self.log_report(f"RMSE          : {_with_ci(m['rmse'], digits=4)}")
            self.log_report(f"Mean Abs Error: {_with_ci(m['mae'], digits=4)}")
            self.log_report(f"Mean Error    : {_with_ci(m['mean_error'], signed=True, digits=4)} (prediction - truth)")
            return

        self._log_intervals(result, sliced=True)
        for name, block in result["groups"].items():
            m = block["metrics"]
            self.log_report(self._group_header(name, block, result))
            self.log_report(f"  RMSE      : {_with_ci(m['rmse'])}")
            self.log_report(f"  MAE       : {_with_ci(m['mae'])}")
            self.log_report(f"  Mean Error: {_with_ci(m['mean_error'], signed=True)} (prediction - truth)")
            self.log_report(f"  R²        : {_with_ci(m['r2'])}\n")

        _grouped_bars(ax, result, [("rmse", "RMSE", "#1f77b4"), ("mean_error", "Mean Error (bias)", "#d62728")])
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_ylabel(f"Error in {self.target_col} units (95% CI)")
        ax.set_title(f"Regression Error by {slice_col}")
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=2)

        gaps = result["gaps"]
        self.log_report("--- DISPARITY SCORES ---")
        self._log_gap(gaps["rmse"], "RMSE Diff", "RMSE Diff", "RMSE differs between groups beyond sampling noise.")
        self._log_gap(gaps["mae"], "MAE Diff", "MAE Diff", "MAE differs between groups beyond sampling noise.")
        self._log_gap(gaps["mean_error"], "Mean Error Diff", "Mean Error Diff",
                      "The model's bias (mean error) differs between groups beyond sampling noise.")

    # ------------------------------------------------------------------
    # Multiclass
    # ------------------------------------------------------------------

    def _render_multiclass_global(self, ax, result):
        self.log_report("--- MULTICLASS GLOBAL REPORT ---", clear=True)

        labels = self.unique_classes
        cm = np.array(result["global"]["confusion"])

        cax = ax.matshow(cm, cmap='Blues')
        self.figure.colorbar(cax)

        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

        m = result["global"]["metrics"]
        self._log_intervals(result, sliced=False)
        self.log_report(f"Accuracy : {_with_ci(m['accuracy'])}")
        self.log_report(f"Bal. Acc.: {_with_ci(m['balanced_accuracy'])} (mean per-class recall)")
        self.log_report(f"Macro F1 : {f1_score(self.y_true, self.y_pred, average='macro', zero_division=0):.3f}\n")

        self.log_report("--- PER-CLASS METRICS ---")
        class_report = classification_report(self.y_true, self.y_pred, labels=labels, target_names=labels, zero_division=0)
        self.log_report(class_report)

    def _render_multiclass_sliced(self, ax, slice_col, result):
        self.log_report(f"--- MULTICLASS FAIRNESS REPORT: {slice_col.upper()} ---", clear=True)
        self._log_intervals(result, sliced=True)
        classes = result["classes"]

        for name, block in result["groups"].items():
            m = block["metrics"]
            self.log_report(self._group_header(name, block, result))
            self.log_report(f"  Accuracy  : {_with_ci(m['accuracy'])}")
            self.log_report(f"  Bal. Acc. : {_with_ci(m['balanced_accuracy'])}")
            for c in classes:
                self.log_report(f"  '{c}': Recall={_with_ci(m[f'recall:{c}'])} | Selected={_with_ci(m[f'selection_rate:{c}'])}")
            self.log_report("")

        _grouped_bars(ax, result, [(f"recall:{c}", f"Recall of '{c}'", None) for c in classes])
        ax.set_ylabel('Recall (95% CI)')
        ax.set_title(f"Per-Class Recall by {slice_col}")
        ax.set_ylim(0, 1.1)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=min(len(classes), 4))

        gaps = result["gaps"]
        self.log_report("--- DISPARITY SCORES ---")
        self._log_gap(gaps["accuracy"], "Accuracy Diff", "Accuracy Diff", "High disparity in Accuracy.")
        self._log_gap(gaps["balanced_accuracy"], "Balanced Accuracy Diff", "Balanced Accuracy Diff",
                      "High disparity in Balanced Accuracy.")
        for c in classes:
            self._log_gap(gaps[f"recall:{c}"], f"Recall Diff ('{c}')", f"Recall Diff ('{c}')",
                          f"High disparity in how often '{c}' cases are recognised.")
        for c in classes:
            self._log_gap(gaps[f"selection_rate:{c}"], f"Selection Diff ('{c}')", f"Selection Diff ('{c}')",
                          f"High disparity in how often '{c}' is predicted.")


def _number(value, signed=False, digits=3):
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _with_ci(metric, signed=False, digits=3):
    """'0.667 [0.333, 1.000]': the value, then its confidence interval when there is one."""
    text = _number(metric["value"], signed, digits)
    if metric.get("ci") is not None:
        lo, hi = metric["ci"]
        text += f" [{_number(lo, signed, digits)}, {_number(hi, signed, digits)}]"
    return text


def _binary_counts(confusion, result):
    """(tn, fp, fn, tp) from a confusion matrix in class order."""
    classes = result["classes"]
    pos = classes.index(result["positive_class"])
    neg = 1 - pos
    cm = np.asarray(confusion)
    return int(cm[neg, neg]), int(cm[neg, pos]), int(cm[pos, neg]), int(cm[pos, pos])


def _grouped_bars(ax, result, series):
    """One bar per group and metric with its confidence interval; small groups are hatched and faded."""
    names = list(result["groups"])
    blocks = [result["groups"][name] for name in names]
    x = np.arange(len(names))
    width = 0.8 / len(series)
    for i, (key, label, color) in enumerate(series):
        heights, below, above = [], [], []
        for block in blocks:
            metric = block["metrics"][key]
            value, ci = metric["value"], metric["ci"]
            if not np.isfinite(value):
                heights.append(0.0); below.append(0.0); above.append(0.0)
                continue
            lo, hi = ci if ci is not None else (value, value)
            heights.append(value); below.append(max(value - lo, 0.0)); above.append(max(hi - value, 0.0))
        bars = ax.bar(x + (i - (len(series) - 1) / 2) * width, heights, width, label=label, color=color,
                      yerr=[below, above], capsize=3)
        for bar, block in zip(bars, blocks):
            if not block["eligible"]:
                bar.set_alpha(0.35)
                bar.set_hatch("//")
    subjects = result["bootstrap"]["unit"] == "group"
    ax.set_xticks(x)
    ax.set_xticklabels([f"{name}\n(n={block['n']}" + (f", {block['units']} subj.)" if subjects else ")")
                        + ("" if block["eligible"] else "\nsmall") for name, block in zip(names, blocks)])
