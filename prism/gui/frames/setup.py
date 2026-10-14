"""
SETUP SCREEN
Handles file picking and threaded ingestion.
Instantiates the Inspector based on framework selection.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from prism.core.subject import load_subject

from prism.gui.utils import create_info_button
from prism.gui.worker import Task

from prism.inspectors import get_inspectors

class SetupFrame(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.inspector_registry = get_inspectors()
        self._init_ui()
    def _init_ui(self):
        config_frame = ttk.LabelFrame(self, text="Connector Configuration", padding=20)
        config_frame.pack(fill=tk.X, padx=20, pady=20)
        
        ttk.Label(config_frame, text="Project Root:").grid(row=0, column=0, sticky="w")
        create_info_button(config_frame, "Project Root", 
            "The master folder containing your AI project.\n\n"
            "PRISM adds this folder to Python's system path so that internal imports "
            "inside your Connector Script will work.").grid(row=0, column=1, padx=(0,5))
        self.ent_root = ttk.Entry(config_frame, width=50)
        self.ent_root.grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(config_frame, text="Browse", command=self.browse_root).grid(row=0, column=3)
        
        ttk.Label(config_frame, text="Connector Script:").grid(row=1, column=0, sticky="w")
        create_info_button(config_frame, "Connector Script",
            "A Python file you write to act as a bridge between PRISM and your code.\n\n"
            "It defines a class that inherits from PrismConnector (from prism import PrismConnector).").grid(row=1, column=1, padx=(0,5))
        self.ent_script = ttk.Entry(config_frame, width=50)
        self.ent_script.grid(row=1, column=2, padx=5, pady=5)
        ttk.Button(config_frame, text="Browse", command=self.browse_script).grid(row=1, column=3)
        
        ttk.Label(config_frame, text="Framework:").grid(row=2, column=0, sticky="w")
        create_info_button(config_frame, "Framework", 
            "The framework your model is built with.\n\n"
            "This tells PRISM which Inspector plugin to use to extract the hidden layers."
        ).grid(row=2, column=1, padx=(0,5))
        
        # Populate dynamically!
        available_frameworks = list(self.inspector_registry.keys())
        self.cb_framework = ttk.Combobox(config_frame, values=available_frameworks, state="readonly")
        if available_frameworks:
            self.cb_framework.current(0)
        self.cb_framework.grid(row=2, column=2, sticky="w", padx=5, pady=5)
        
        self.btn_ingest = ttk.Button(self, text="INGEST SUBJECT", command=self.run_ingest)
        self.btn_ingest.pack(pady=10)
        self.progress = ttk.Progressbar(self, mode="indeterminate", length=300)
        self.progress.pack()
        self.lbl_status = ttk.Label(self, text="")
        self.lbl_status.pack(pady=5)

    def browse_root(self):
        path = filedialog.askdirectory()
        if path:
            self.ent_root.delete(0, tk.END)
            self.ent_root.insert(0, path)

    def browse_script(self):
        path = filedialog.askopenfilename(filetypes=[("Python", "*.py")])
        if path:
            self.ent_script.delete(0, tk.END)
            self.ent_script.insert(0, path)

    def run_ingest(self):
        script = self.ent_script.get()
        if not script:
            messagebox.showinfo("No Connector Script", "Choose a connector script first.")
            return
        self.ingest(self.ent_root.get(), script, self.cb_framework.get())

    def ingest(self, root, script, fw, then=None):
        """Load the connector, model and inspector off the Tk thread, then call `then()` if given;
        returns the running Task."""
        self.btn_ingest.config(state="disabled")
        self.lbl_status.config(text="Loading connector and model...", foreground="black")
        self.progress.start(15)
        registry = self.inspector_registry

        def work(task):
            subject = load_subject(root, script, fw, registry=registry)
            hidden = getattr(subject.inspector, 'extracts_hidden_layers', True)
            if hidden:
                task.status("Discovering layers...")
            return subject, (subject.inspector.discover_topology() if hidden else None)

        def done(outcome):
            subject, layers = outcome
            app = self.app
            app.connector_instance, app.model_subject, app.inspector = subject.connector, subject.model, subject.inspector
            app.project_root, app.script_path, app.framework = root, script, fw
            if layers is None:
                app.log("Black-box framework selected. Bypassing Wiretap.")
                app.frame_wiretap.populate_tree([])
                self.lbl_status.config(text="Loaded (black-box model: no layers to choose).", foreground="green")
            else:
                app.log(f"Discovered {len(layers)} layers.")
                app.frame_wiretap.populate_tree(layers)
                self.lbl_status.config(text=f"Loaded: {len(layers)} layers found.", foreground="green")
            app.set_subject_loaded("Loaded")
            if then is not None:
                then()

        def failed(error, trace):
            self.app.log(trace)
            self.lbl_status.config(text="Ingestion failed.", foreground="red")
            messagebox.showerror("Ingestion failed", f"Could not load the connector or model:\n\n{error}")

        def finish():
            self.progress.stop()
            self.btn_ingest.config(state="normal")

        return Task(self, work, on_done=done, on_error=failed, on_finish=finish,
                    on_status=lambda text: self.lbl_status.config(text=text)).start()
