"""
EVIDENCE SCREEN
Loads the metadata CSV, maps the columns to their roles, and performs
an automated "Leakage/Confound" check to warn users of bad experimental design.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from prism.analysis import confound_screen
from prism.core.metadata import read_metadata
from prism.gui.utils import create_info_button

RISK_TAGS = {"EXTREME": "danger", "HIGH": "warn", "MEDIUM": "warn"}

class EvidenceFrame(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.df = None # Holds the Pandas DataFrame
        self.csv_path = None

        self._init_ui()

    def _init_ui(self):
        paned = ttk.PanedWindow(self, orient=tk.VERTICAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        top_frame = ttk.Frame(paned); bottom_frame = ttk.Frame(paned)
        paned.add(top_frame, weight=1); paned.add(bottom_frame, weight=2)
        
        ctrl_frame = ttk.LabelFrame(top_frame, text="1. Load & Map Metadata", padding=10)
        ctrl_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        ttk.Label(ctrl_frame, text="Metadata CSV:").grid(row=0, column=0, sticky="w", pady=5)
        create_info_button(ctrl_frame, "Metadata CSV", "The table containing your test dataset labels and demographics.").grid(row=0, column=1, padx=(0,5))
        self.ent_csv = ttk.Entry(ctrl_frame, width=30)
        self.ent_csv.grid(row=0, column=2, padx=5)
        ttk.Button(ctrl_frame, text="Browse", command=self.browse_csv).grid(row=0, column=3)
        
        ttk.Label(ctrl_frame, text="ID Column:").grid(row=1, column=0, sticky="w", pady=5)
        create_info_button(ctrl_frame, "ID Column", "The unique identifier (e.g. filename) passed to your Connector Script.").grid(row=1, column=1, padx=(0,5))
        self.cb_id = ttk.Combobox(ctrl_frame, state="readonly")
        self.cb_id.bind("<<ComboboxSelected>>", self.on_id_selected)
        self.cb_id.grid(row=1, column=2, columnspan=2, sticky="ew", padx=5)
        
        ttk.Label(ctrl_frame, text="Ground Truth:").grid(row=2, column=0, sticky="w", pady=5)
        create_info_button(ctrl_frame, "Ground Truth", "The target the model is attempting to predict.").grid(row=2, column=1, padx=(0,5))
        self.cb_target = ttk.Combobox(ctrl_frame, state="readonly")
        self.cb_target.grid(row=2, column=2, columnspan=2, sticky="ew", padx=5)
        self.cb_target.bind("<<ComboboxSelected>>", self.run_leakage_check)
        
        ttk.Label(ctrl_frame, text="Group By:").grid(row=3, column=0, sticky="w", pady=5)
        create_info_button(ctrl_frame, "Group By (Leakage Prevention)", "Crucial! Select 'Patient_ID' to ensure scans from the same patient are never split between Training and Testing sets during Probing.").grid(row=3, column=1, padx=(0,5))
        self.cb_group = ttk.Combobox(ctrl_frame, state="readonly")
        self.cb_group.bind("<<ComboboxSelected>>", self.run_leakage_check)  # grouping changes the screen
        self.cb_group.grid(row=3, column=2, columnspan=2, sticky="ew", padx=5)
        
        ttk.Button(ctrl_frame, text="CONFIRM MAPPING", command=self.confirm_mapping).grid(row=4, column=0, columnspan=4, pady=15)

        # ==========================================
        # TOP RIGHT: METADATA PROFILER & LEAKAGE
        # ==========================================
        prof_frame = ttk.LabelFrame(top_frame, text="2. Confound & Leakage Analysis", padding=10)
        prof_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        cols = ("Feature", "Type", "Unique", "Missing", "Leakage Risk")
        self.tree_prof = ttk.Treeview(prof_frame, columns=cols, show="headings", height=8)
        for c in cols:
            self.tree_prof.heading(c, text=c)
            self.tree_prof.column(c, width=80)
        self.tree_prof.column("Feature", width=120)
        self.tree_prof.column("Leakage Risk", width=150)
        
        self.tree_prof.pack(fill=tk.BOTH, expand=True)

        # ==========================================
        # BOTTOM: DATA PREVIEW
        # ==========================================
        prev_frame = ttk.LabelFrame(bottom_frame, text="Data Preview (First 50 Rows)", padding=10)
        prev_frame.pack(fill=tk.BOTH, expand=True)
        
        self.tree_prev = ttk.Treeview(prev_frame, show="headings")
        sb_x = ttk.Scrollbar(prev_frame, orient="horizontal", command=self.tree_prev.xview)
        sb_y = ttk.Scrollbar(prev_frame, orient="vertical", command=self.tree_prev.yview)
        self.tree_prev.configure(xscrollcommand=sb_x.set, yscrollcommand=sb_y.set)
        
        self.tree_prev.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb_y.pack(side=tk.RIGHT, fill=tk.Y)
        sb_x.pack(side=tk.BOTTOM, fill=tk.X)

    # --- ACTIONS ---

    def browse_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if path:
            self.ent_csv.delete(0, tk.END)
            self.ent_csv.insert(0, path)
            self.load_csv(path)

    def load_csv(self, path):
        try:
            self.df = read_metadata(path)
            self.csv_path = path
            columns = list(self.df.columns)
            
            # Populate dropdowns
            self.cb_id['values'] = columns
            self.cb_target['values'] = columns
            self.cb_group['values'] = ["(None)"] + columns
            self.cb_group.current(0)
            
            # Auto-guess ID and Target if named obviously
            if "id" in columns[0].lower():
                self.cb_id.current(0)
                self.df = read_metadata(path, columns[0])

            self.update_preview()
            self.run_leakage_check() # Initial profile
            
            self.app.log(f"Loaded CSV: {len(self.df)} rows, {len(columns)} columns.")
            self.app.lbl_evidence.config(text="Evidence: Loaded (Unmapped)", foreground="orange")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV:\n{e}")

    def on_id_selected(self, event=None):
        """Re-read the CSV with the chosen ID column as text, so IDs such as 0042 keep their zeros,
        then refresh the preview and the screen (which never screens the ID column)."""
        if self.csv_path:
            self.df = read_metadata(self.csv_path, self.cb_id.get() or None)
            self.update_preview()
        self.run_leakage_check()

    def update_preview(self):
        # Clear old preview
        self.tree_prev.delete(*self.tree_prev.get_children())
        
        # Set new columns
        self.tree_prev["columns"] = list(self.df.columns)
        for col in self.df.columns:
            self.tree_prev.heading(col, text=col)
            self.tree_prev.column(col, width=100)
            
        # Insert first 50 rows
        for _, row in self.df.head(50).iterrows():
            self.tree_prev.insert("", "end", values=list(row))

    def run_leakage_check(self, event=None):
        """
        Profiles the dataset. Once a Ground Truth is chosen, every other column is
        screened as a possible proxy for it (analysis.confound_screen): the AUC shown is
        how well that column alone predicts the target, 0.5 meaning not at all.
        """
        if self.df is None: return
        
        self.tree_prof.delete(*self.tree_prof.get_children())
        target_col = self.cb_target.get()

        screened = {}
        if target_col and target_col in self.df.columns:
            id_col = self.cb_id.get()
            group_col = self.cb_group.get()
            group_col = group_col if group_col in self.df.columns else None  # "(None)" or unset
            screened = {row["field"]: row for row in confound_screen(
                self.df, target_col, id_cols=[id_col] if id_col else [], group_col=group_col)}
        
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            unique_count = self.df[col].nunique()
            missing_pct = f"{(self.df[col].isna().mean() * 100):.1f}%"

            row = screened.get(col)
            if row is None or row["auc"] is None:
                leakage_risk, risk_tag = "N/A", "normal"
            else:
                leakage_risk = f"{row['risk']} (AUC {row['auc']:.2f})"
                risk_tag = RISK_TAGS.get(row["risk"], "normal")

            # Insert into profiler tree
            self.tree_prof.insert("", "end", values=(col, dtype, unique_count, missing_pct, leakage_risk), tags=(risk_tag,))
            
        # Add colors to tags
        self.tree_prof.tag_configure("danger", foreground="red")
        self.tree_prof.tag_configure("warn", foreground="orange")

    def confirm_mapping(self, quiet=False):
        if not self.cb_id.get() or not self.cb_target.get():
            messagebox.showwarning("Incomplete", "You must select an ID column and a Ground Truth column.")
            return
            
        # Save to global app state, with the ID column as text whichever way it was chosen
        if self.csv_path:
            self.df = read_metadata(self.csv_path, self.cb_id.get())
        self.app.metadata_df = self.df
        self.app.csv_path = self.csv_path
        self.app.config_id_col = self.cb_id.get()
        self.app.config_target_col = self.cb_target.get()
        self.app.config_group_col = self.cb_group.get() if self.cb_group.get() != "(None)" else None
        
        self.app.set_evidence_loaded()
        self.app.log("Evidence Mapping Confirmed. Ready for Execution.")
        if not quiet:  # an opened setup confirms without a dialog
            messagebox.showinfo("Success", "Data mapped successfully. Proceed to Execution Tab.")