"""
ERROR GALLERY
Visualizes specific predictions (False Positives/Negatives) 
and finds mathematically similar images in the embedding space.
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from prism.core.labels import class_order, decode_predictions
from prism.gui.utils import column_mapping, open_cache
from prism.gui.worker import Runner

NO_LAYERS = "These results have no layers to find neighbours in."


class GalleryFrame(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.canvas_suspect = None
        self.canvas_neighbors = None
        self.cache = None
        self.id_col = None
        self._embedding_key, self._embedding = None, None  # the search-space layer, kept between clicks
        self.runner = Runner(self)  # items load off the Tk thread, one inspection at a time
        self._generation = 0        # bumped by load_data, so an inspection of older results is not drawn
        self._init_ui()

    def _init_ui(self):
        # --- TOP CONTROLS ---
        ctrl_frame = ttk.Frame(self)
        ctrl_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(ctrl_frame, text="Load Data", command=self.load_data).grid(row=0, column=0, padx=5)
        
        ttk.Label(ctrl_frame, text="Filter:").grid(row=0, column=1, padx=5)
        self.cb_filter = ttk.Combobox(ctrl_frame, state="readonly", values=["All", "Errors Only", "False Positives", "False Negatives"])
        self.cb_filter.current(0)
        self.cb_filter.grid(row=0, column=2, padx=5)
        self.cb_filter.bind("<<ComboboxSelected>>", self.populate_list)

        ttk.Label(ctrl_frame, text="Search Space (Layer):").grid(row=0, column=3, padx=20)
        self.cb_layer = ttk.Combobox(ctrl_frame, state="readonly")
        self.cb_layer.grid(row=0, column=4, padx=5)

        # What to know about these results (rows left out, guessed class order); empty when nothing
        self.lbl_notes = ttk.Label(self, text="", foreground="#a6510b", wraplength=1000, justify="left")
        self.lbl_notes.pack(fill=tk.X, padx=10)

        # --- SPLIT LAYOUT ---
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # LEFT: The List of Images
        list_frame = ttk.LabelFrame(paned, text="Dataset (Click to inspect)")
        paned.add(list_frame, weight=1)
        
        cols = ("ID", "Truth", "Pred", "Status")
        self.tree = ttk.Treeview(list_frame, columns=cols, show="headings", selectmode="browse")
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=70)
        self.tree.column("ID", width=120)
        
        sb = ttk.Scrollbar(list_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.tree.bind("<<TreeviewSelect>>", self.on_select_item)

        # RIGHT: The Inspection Panel
        right_paned = ttk.PanedWindow(paned, orient=tk.VERTICAL)
        paned.add(right_paned, weight=3)

        # Top Right: The Suspect
        self.frm_suspect = ttk.LabelFrame(right_paned, text="The Suspect")
        right_paned.add(self.frm_suspect, weight=1)

        # Bottom Right: The Lineup (Nearest Neighbors)
        self.frm_lineup = ttk.LabelFrame(right_paned, text="The Lineup (Nearest Neighbors in Representation Space)")
        right_paned.add(self.frm_lineup, weight=1)

    def load_data(self):
        if not self.app.connector_instance:
            messagebox.showerror("Error", "No Connector Loaded. Go to Setup.")
            return
        cache = open_cache(self.app)
        if cache is None: return
        self.runner.cancel()
        self._generation += 1
        self.cache = cache
        self._embedding_key = None
        self.df = cache.metadata
        self.raw_preds = cache.model_output()
        layers = cache.layer_names
        self.cb_layer['values'] = layers
        if layers: self.cb_layer.current(0)

        self.id_col, target_col, _ = column_mapping(self.app, cache)
        target_series = self.df[target_col]
        
        # If it's a float, or if it's an int with many unique values, block it.
        if pd.api.types.is_float_dtype(target_series) or (pd.api.types.is_numeric_dtype(target_series) and target_series.nunique() > 10):
            messagebox.showinfo("V1 Limitation", "The Error Gallery does not currently support Regression tasks.")
            return
        if target_series.nunique() > 2:
            messagebox.showinfo("V1 Limitation", "The Error Gallery currently only supports Binary Classification tasks (2 classes).")
            return
        
        
        self.y_true = target_series.astype(str).values

        # Same decoding as the Performance tab, so both tabs agree on every prediction
        self.class_order, notes = class_order(target_series, cache.class_names)
        self.lbl_notes.config(text="\n".join(f"Note: {note}" for note in cache.warnings + notes))
        try:
            self.y_pred = decode_predictions(self.raw_preds, self.class_order, cache.output_kind)
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return

        self.populate_list()

    def _embeddings(self, layer):
        """The layer's rows as a float32 matrix with their squared norms, loaded once per layer."""
        key = (self.cache.output_dir, layer)
        if self._embedding_key != key:
            X = self.cache.layer(layer)
            X = np.ascontiguousarray(X.reshape(len(X), -1), dtype=np.float32)
            self._embedding, self._embedding_key = (X, np.einsum("ij,ij->i", X, X)), key
        return self._embedding

    def row_id(self, idx):
        """The ID the connector was given for cache row `idx` during extraction."""
        if not self.cache.legacy:
            return self.cache.ids[idx]
        if self.id_col in self.df.columns:
            return str(self.df.iloc[idx][self.id_col])
        return str(idx)

    def populate_list(self, event=None):
        self.tree.delete(*self.tree.get_children())
        filter_val = self.cb_filter.get()

        # The positive class is the last in model-output order (what a single score measures)
        pos_class = self.class_order[-1]

        for idx in range(len(self.df)):
            truth = self.y_true[idx]
            pred = self.y_pred[idx]
            
            status = "Correct"
            if truth != pred:
                if pred == pos_class: status = "False Pos"
                else: status = "False Neg"
                
            # Apply Filter
            if filter_val == "Errors Only" and status == "Correct": continue
            if filter_val == "False Positives" and status != "False Pos": continue
            if filter_val == "False Negatives" and status != "False Neg": continue
            
            # Store the actual index (idx) in the 'tags' so we can look it up later!
            self.tree.insert("", "end", values=(self.row_id(idx), truth, pred, status), tags=(str(idx),))

    def on_select_item(self, event):
        selected = self.tree.selection()
        if not selected: return

        # Grab the dataset index we stored in the tag
        idx = int(self.tree.item(selected[0], "tags")[0])
        return self.inspect(idx)

    def inspect(self, idx):
        """Load the item and its nearest neighbours in the background (the connector may be slow,
        e.g. MRI volumes), then draw them. Returns the Task."""
        layer, group_col, generation = self.cb_layer.get(), self._group_col(), self._generation
        self._message(self.frm_suspect, "Loading...")
        self._message(self.frm_lineup, "Loading..." if layer else NO_LAYERS)

        def work(task):
            suspect = self._fetch_item(idx)
            return suspect, (self._fetch_neighbors(idx, layer, group_col, task) if layer else None)

        def done(outcome):
            if generation != self._generation:
                return
            suspect, neighbours = outcome
            self._show_suspect(idx, suspect)
            if neighbours is not None:
                self._show_neighbors(neighbours)

        def failed(error, trace):
            print(trace)
            self._message(self.frm_suspect, "")
            self._message(self.frm_lineup, "")
            messagebox.showerror("Could not load the item", str(error))

        return self.runner.submit(work, on_done=done, on_error=failed)

    def _render_suspect(self, idx):
        self._show_suspect(idx, self._fetch_item(idx))

    def _render_neighbors(self, suspect_idx):
        layer = self.cb_layer.get()
        if not layer:
            self._message(self.frm_lineup, NO_LAYERS)
            return
        self._show_neighbors(self._fetch_neighbors(suspect_idx, layer, self._group_col()))

    def _group_col(self):
        _, _, group_col = column_mapping(self.app, self.cache)
        return group_col if group_col and group_col in self.df.columns else None

    @staticmethod
    def _message(frame, text):
        for widget in frame.winfo_children(): widget.destroy()
        if text:
            ttk.Label(frame, text=text).pack()

    # --- background half: the connector loads and renders, nothing touches Tk ----------

    def _fetch_item(self, idx):
        """Ask the user's code for the item and for how to draw it."""
        metadata_row = self.df.iloc[idx].to_dict()
        raw_data = self.app.connector_instance.load_item(self.row_id(idx), metadata_row)
        return self.app.connector_instance.render_item_for_gallery(raw_data)

    def _fetch_neighbors(self, suspect_idx, layer, group_col, task=None):
        """[(row, distance, render dict)] for the 3 nearest rows outside the suspect's Group By group."""
        X, squared_norms = self._embeddings(layer)

        # Candidates: everyone except the suspect and, with Group By, the suspect's own group
        # (otherwise the lineup is mostly the same subject's other scans).
        excluded = np.zeros(len(X), dtype=bool)
        if group_col:
            groups = self.df[group_col]
            excluded = (groups == groups.iloc[suspect_idx]).to_numpy()
        excluded[suspect_idx] = True
        candidates = np.flatnonzero(~excluded)
        if len(candidates) == 0:
            return []

        # Nearest neighbours by Euclidean distance, computed directly (no index to rebuild per click)
        x = X[suspect_idx]
        squared = np.clip(squared_norms[candidates] - 2 * (X[candidates] @ x) + x @ x, 0, None)
        k = min(3, len(candidates))
        nearest = np.argpartition(squared, k - 1)[:k] if k < len(candidates) else np.arange(len(candidates))
        nearest = nearest[np.lexsort((candidates[nearest], squared[nearest]))]  # by distance, then row

        lineup = []
        for n_idx, distance in zip(candidates[nearest], np.sqrt(squared[nearest])):
            if task is not None:
                task.check()
            lineup.append((int(n_idx), float(distance), self._fetch_item(n_idx)))
        return lineup

    # --- Tk half: drawing -----------------------------------------------------------------

    def _show_suspect(self, idx, render_dict):
        # Clear old drawing
        for widget in self.frm_suspect.winfo_children(): widget.destroy()

        if render_dict['type'] == 'image':
            fig = Figure(figsize=(3, 3))  # not pyplot, which would keep every figure alive
            ax = fig.add_subplot()
            img_array = render_dict['payload']
            # Assume grayscale if 2D, RGB if 3D
            if len(img_array.shape) == 2:
                ax.imshow(img_array, cmap='gray')
            else:
                # Assuming PyTorch format (C, H, W). Move to (H, W, C) for plotting
                img_array = np.transpose(img_array, (1, 2, 0))
                ax.imshow(img_array)

            ax.axis('off')
            ax.set_title(f"Target: {self.y_true[idx]} | Pred: {self.y_pred[idx]}")

            canvas = FigureCanvasTkAgg(fig, master=self.frm_suspect)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        elif render_dict['type'] == 'table':
            _table(self.frm_suspect, render_dict['payload']).pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        else:  # 'text', and anything else shown as its text
            txt = tk.Text(self.frm_suspect, wrap="word", font=("Arial", 11))
            txt.insert("1.0", str(render_dict['payload']))
            txt.config(state="disabled")
            txt.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _show_neighbors(self, lineup):
        # Clear old layout
        for widget in self.frm_lineup.winfo_children(): widget.destroy()
        if not lineup:
            ttk.Label(self.frm_lineup, text="No other groups to compare with.").pack()
            return

        # Draw the 3 neighbors horizontally
        for n_idx, distance, r_dict in lineup:
            frame = ttk.Frame(self.frm_lineup)
            frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

            if r_dict['type'] == 'image':
                fig = Figure(figsize=(2, 2))
                ax = fig.add_subplot()
                img = r_dict['payload']
                if len(img.shape) == 2: ax.imshow(img, cmap='gray')
                else: ax.imshow(np.transpose(img, (1, 2, 0)))
                ax.axis('off')

                truth = self.y_true[n_idx]
                ax.set_title(f"Truth: {truth}\nDist: {distance:.2f}", fontsize=8)

                canvas = FigureCanvasTkAgg(fig, master=frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            elif r_dict['type'] == 'table':
                ttk.Label(frame, text=f"Truth: {self.y_true[n_idx]} | Dist: {distance:.2f}", font="bold").pack()
                _table(frame, r_dict['payload'], height=5).pack(fill=tk.BOTH, expand=True)
            else:
                ttk.Label(frame, text=f"Truth: {self.y_true[n_idx]}", font="bold").pack()
                txt = tk.Text(frame, height=5, width=20, font=("Arial", 9))
                txt.insert("1.0", str(r_dict['payload'])[:100] + "...")
                txt.pack()


def _table(master, payload, height=10):
    """A {field: value} payload as a two-column list."""
    tree = ttk.Treeview(master, columns=("field", "value"), show="headings", height=height)
    tree.heading("field", text="Field")
    tree.heading("value", text="Value")
    tree.column("field", width=120)
    tree.column("value", width=160)
    rows = payload.items() if isinstance(payload, dict) else enumerate(payload)
    for field, value in rows:
        tree.insert("", "end", values=(str(field), str(value)))
    return tree
