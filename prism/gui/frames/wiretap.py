"""
WIRETAP SCREEN
Lists the model's layers. The user ticks the layers to extract and sets each one's pooling,
and can run one row through the model to see what each layer outputs and what its pooling keeps.

Ticking (what gets extracted) and selecting (which rows a pooling change applies to) are
separate, so layers can get different poolings without rebuilding the choice of layers.
"""
import tkinter as tk
from math import prod
from tkinter import messagebox, ttk

from prism.core.config import DEFAULT_POOLING, POOLING_STRATEGIES, shape_text, stored_shape
from prism.core.execution import preview_shapes
from prism.core.cache import module_of
from prism.core.subject import Subject
from prism.gui.utils import create_info_button
from prism.gui.worker import Task

TICKED, UNTICKED = "☑", "☐"

POOLING_HELP = (
    "Mean Pool: each channel's average. CNNs.\n"
    "Max Pool: each channel's strongest value. CNNs.\n"
    "Grid Pool: each channel's average on a 4×4 grid, keeping rough position. CNNs.\n"
    "Seq Pool: the average over tokens. Transformers, RNNs.\n"
    "CLS Token: the first token. BERT-style transformers.\n"
    "Flatten: every value. Large.")


class WiretapFrame(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.shapes = {}   # {layer: [shape of one row's output, per call]} from Show shapes
        self.task = None   # the Show shapes run
        self._init_ui()

    def _init_ui(self):
        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=10, pady=(10, 0))
        hint_text = "Tick layers to extract. Right-click to set pooling."
        ttk.Label(top, text=hint_text, font=("Arial", 12, "bold")).pack(side=tk.LEFT)
        create_info_button(top, "Pooling", POOLING_HELP).pack(side=tk.LEFT, padx=5)

        actions = ttk.Frame(self)
        actions.pack(fill=tk.X, padx=10, pady=(6, 0))
        self.btn_shapes = ttk.Button(actions, text="Show shapes", command=self.show_shapes)
        self.btn_shapes.pack(side=tk.LEFT)
        ttk.Button(actions, text="Tick selected", command=lambda: self.set_ticked(True)).pack(side=tk.LEFT, padx=5)
        ttk.Button(actions, text="Untick selected", command=lambda: self.set_ticked(False)).pack(side=tk.LEFT)
        self.lbl_status = ttk.Label(actions, text="")
        self.lbl_status.pack(side=tk.LEFT, padx=10)

        cols = ("Extract", "Type", "Params", "Pooling", "Output", "Stored")
        self.tree = ttk.Treeview(self, columns=cols, selectmode="extended")
        self.tree.heading("#0", text="Layer Name", anchor="w")
        headings = {"Output": "Output (one row)", "Stored": "Stored values"}
        widths = {"Extract": 60, "Type": 100, "Params": 90, "Pooling": 90, "Output": 140, "Stored": 140}
        for col in cols:
            self.tree.heading(col, text=headings.get(col, col), anchor="w")
            self.tree.column(col, width=widths[col], anchor="center" if col == "Extract" else "w")
        self.tree.column("#0", width=260)

        sb = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.context_menu = tk.Menu(self, tearoff=0)
        for entry in [
                ("Mean Pool (CNNs)", "Mean Pool"),
                ("Max Pool (CNNs)", "Max Pool"),
                ("Grid Pool (CNNs)", "Grid Pool"),
                None,
                ("Seq Pool (transformers, RNNs)", "Seq Pool"),
                ("CLS Token (BERT-style)", "CLS Token"),
                None,
                ("Flatten (large)", "Flatten")]:
            if entry is None:
                self.context_menu.add_separator()
            else:
                label, strategy = entry
                self.context_menu.add_command(label=label, command=lambda s=strategy: self.set_strategy(s))
        self.tree.bind("<Button-1>", self.on_click, add="+")
        self.tree.bind("<space>", lambda event: self.toggle_selected())
        self.tree.bind("<Button-3>", self.show_context_menu)  # Right click on Windows/Linux
        self.tree.bind("<Button-2>", self.show_context_menu)  # Right click on Mac
        self.tree.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        sb.pack(side="right", fill="y", pady=10)

    def populate_tree(self, layers):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.shapes = {}
        self.lbl_status.config(text="")
        for layer in layers:
            self.tree.insert("", "end", text=layer['id'],
                             values=(UNTICKED, layer['type'], layer['params'], DEFAULT_POOLING, "", ""))

    # --- what gets extracted ---------------------------------------------

    def extracted_layers(self):
        """{layer name: pooling strategy} for the ticked layers, in model order."""
        return {self.tree.item(item, "text"): self.tree.set(item, "Pooling")
                for item in self.tree.get_children() if self.tree.set(item, "Extract") == TICKED}

    def tick_layers(self, layers):
        """Tick {layer name: pooling} with their pooling (an opened setup); returns the names not found."""
        items = {self.tree.item(item, "text"): item for item in self.tree.get_children()}
        for name, pooling in layers.items():
            if name in items:
                self._tick(items[name], True)
                self.tree.set(items[name], "Pooling", pooling if pooling in POOLING_STRATEGIES else DEFAULT_POOLING)
                self._show_stored(items[name])
        return [name for name in layers if name not in items]

    def on_click(self, event):
        """A click on a row's box ticks or unticks that row, and leaves the selection alone."""
        if self.tree.identify_region(event.x, event.y) == "cell" and self.tree.identify_column(event.x) == "#1":
            item = self.tree.identify_row(event.y)
            if item:
                self._tick(item, self.tree.set(item, "Extract") != TICKED)
                return "break"
        return None

    def toggle_selected(self):
        """Space: tick the selected rows, or untick them if they are all ticked already."""
        selected = self.tree.selection()
        self.set_ticked(not all(self.tree.set(item, "Extract") == TICKED for item in selected))
        return "break"

    def set_ticked(self, ticked):
        for item in self.tree.selection():
            self._tick(item, ticked)

    def _tick(self, item, ticked):
        self.tree.set(item, "Extract", TICKED if ticked else UNTICKED)

    # --- pooling -----------------------------------------------------------

    def show_context_menu(self, event):
        item = self.tree.identify_row(event.y)
        if item:
            # Right-clicking inside the selection keeps it (to set the pooling of all of it);
            # right-clicking elsewhere selects just that row. Neither changes what is ticked.
            if item not in self.tree.selection():
                self.tree.selection_set(item)
            self.context_menu.post(event.x_root, event.y_root)

    def set_strategy(self, strategy):
        for item in self.tree.selection():
            self.tree.set(item, "Pooling", strategy)
            self._show_stored(item)

    # --- shapes ------------------------------------------------------------

    def show_shapes(self):
        """Run the first metadata row through the model and show every layer's output shape."""
        app = self.app
        names = [self.tree.item(item, "text") for item in self.tree.get_children()]
        if app.connector_instance is None:
            messagebox.showerror("No Subject Loaded", "Ingest a subject in the Setup tab first.")
            return
        if not names:
            messagebox.showinfo("No Layers", "This model has no layers to choose: PRISM stores only its output.")
            return
        if app.metadata_df is None or not app.config_id_col:
            messagebox.showinfo("Metadata Needed", "Confirm the metadata in the Evidence tab first: Show shapes "
                                                   "runs its first row through the model.")
            return
        extraction = getattr(app.frame_execution, "task", None)
        if (extraction is not None and extraction.running) or (self.task is not None and self.task.running):
            messagebox.showinfo("Busy", "Wait for the running extraction or Show shapes to finish.")
            return

        subject = Subject(app.connector_instance, app.model_subject, app.inspector)
        metadata, id_col = app.metadata_df, app.config_id_col
        first_id = str(metadata[id_col].iloc[0]) if len(metadata) and id_col in metadata else "?"

        def done(shapes):
            self.shapes = {}
            for name, shape in shapes.items():
                self.shapes.setdefault(module_of(name), []).append(shape)
            for item in self.tree.get_children():
                self.tree.set(item, "Output", self._output_text(self.tree.item(item, "text")))
                self._show_stored(item)
            silent = sum(1 for name in names if name not in self.shapes)
            self.lbl_status.config(text=f"Shapes from row {first_id}." + (
                f" {silent} layer{'s' if silent != 1 else ''} did not run for it." if silent else ""))

        def failed(error, trace):
            self.app.log(trace)
            self.lbl_status.config(text="Show shapes failed.")
            messagebox.showerror("Show Shapes Failed", f"Running row {first_id} through the model failed:\n{error}")

        self.btn_shapes.config(state="disabled")
        self.lbl_status.config(text=f"Running row {first_id} through the model...")
        self.task = Task(self, lambda task: preview_shapes(subject, metadata, id_col, names), on_done=done,
                         on_error=failed, on_finish=lambda: self.btn_shapes.config(state="normal")).start()
        return self.task

    def _output_text(self, name):
        calls = self.shapes.get(name)
        if not calls:
            return "did not run" if self.shapes else ""
        unique = list(dict.fromkeys(calls))
        if len(calls) == 1:
            return shape_text(calls[0])
        return (f"{len(calls)} calls × {shape_text(unique[0])}" if len(unique) == 1
                else "; ".join(shape_text(shape) for shape in calls))

    def _show_stored(self, item):
        calls = self.shapes.get(self.tree.item(item, "text"))
        if not calls:
            self.tree.set(item, "Stored", "")
            return
        pooling = self.tree.set(item, "Pooling")
        sizes = []
        for shape in calls:
            stored = stored_shape(shape, pooling)
            sizes.append(f"{prod(stored):,}" + (f" ({shape_text(stored)})" if len(stored) > 1 else ""))
        unique = list(dict.fromkeys(sizes))
        self.tree.set(item, "Stored", unique[0] if len(sizes) == 1 else
                      (f"{len(sizes)} × {unique[0]}" if len(unique) == 1 else "; ".join(sizes)))
