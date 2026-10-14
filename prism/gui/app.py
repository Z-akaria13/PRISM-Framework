"""
MAIN APPLICATION WINDOW
Acts as the central State Manager. Holds references to the loaded connector,
model, and inspector, making them accessible to child frames.
"""
import importlib.util
import tkinter as tk
from tkinter import ttk


from .frames.setup import SetupFrame
from .frames.wiretap import WiretapFrame
from .frames.evidence import EvidenceFrame 
from .frames.execution import ExecutionFrame 
from .frames.performance import PerformanceFrame
from .frames.probing import ProbingFrame
from .frames.gallery import GalleryFrame
from .setup_file import open_setup, save_setup
from .worker import Task
from prism import __version__

class PrismApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"PRISM // v{__version__}")
        self.geometry("1100x750")
        
        # Global State, filled in by the tabs as the user works through them
        # Setup tab
        self.connector_instance = None
        self.model_subject = None
        self.inspector = None
        self.project_root = None
        self.script_path = None
        self.framework = None
        # Evidence tab
        self.metadata_df = None
        self.csv_path = None
        self.config_id_col = None
        self.config_target_col = None
        self.config_group_col = None

        # Styling
        style = ttk.Style()
        style.theme_use('clam')
        self.active_cache_dir = None
        # Layout
        self.paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True)
        
        self._init_menu()
        self._init_sidebar()
        self._init_notebook()
        
    def _init_menu(self):
        menu = tk.Menu(self)
        file_menu = tk.Menu(menu, tearoff=0)
        file_menu.add_command(label="Open Setup...", command=lambda: open_setup(self))
        file_menu.add_command(label="Save Setup...", command=lambda: save_setup(self))
        menu.add_cascade(label="File", menu=file_menu)
        self.config(menu=menu)

    def _init_sidebar(self):
        self.sidebar = ttk.Frame(self.paned, width=200, relief=tk.SUNKEN)
        self.paned.add(self.sidebar, weight=1)
        
        ttk.Label(self.sidebar, text="STATUS", font=("Helvetica", 12, "bold")).pack(pady=10)
        
        self.lbl_subject = ttk.Label(self.sidebar, text="No Subject", foreground="red")
        self.lbl_subject.pack(anchor="w", padx=10, pady=2)
        
        self.lbl_evidence = ttk.Label(self.sidebar, text="No Evidence", foreground="red")
        self.lbl_evidence.pack(anchor="w", padx=10, pady=2)

        self.lbl_setup = ttk.Label(self.sidebar, text="", foreground="gray")  # the setup file opened or saved
        self.lbl_setup.pack(anchor="w", padx=10, pady=2)



        self.lbl_device = ttk.Label(self.sidebar, text="⚪ Checking Device...", foreground="gray")
        self.lbl_device.pack(anchor="w", padx=10, pady=2)
    
        self.check_hardware()
    
    def check_hardware(self):
        """Show whether this machine has a GPU, without holding up the window: importing torch takes
        seconds, and scikit-learn audits don't need it at all."""
        if importlib.util.find_spec("torch") is None:
            self.lbl_device.config(text="PyTorch not installed (CPU)", foreground="gray")
            self._device_task = None
            return

        def work(task):
            import torch
            return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None

        def show(device_name):
            if device_name:
                self.lbl_device.config(text=f"GPU: {device_name}", foreground="green")
            else:
                self.lbl_device.config(text="Running on CPU", foreground="orange")

        self._device_task = Task(self, work, on_done=show, on_error=lambda error, trace: self.lbl_device.config(
            text="Device check failed", foreground="gray")).start()

    def _init_notebook(self):
        self.notebook = ttk.Notebook(self.paned)
        self.paned.add(self.notebook, weight=4)
        
        self.frame_setup = SetupFrame(self.notebook, self)
        self.frame_evidence = EvidenceFrame(self.notebook, self)
        self.frame_wiretap = WiretapFrame(self.notebook, self)
        self.frame_execution = ExecutionFrame(self.notebook, self)
        self.frame_performance = PerformanceFrame(self.notebook, self)
        self.frame_probing = ProbingFrame(self.notebook, self)
        self.frame_gallery = GalleryFrame(self.notebook, self)
        
        self.notebook.add(self.frame_setup, text="1. Setup")
        self.notebook.add(self.frame_evidence, text="2. Evidence")
        self.notebook.add(self.frame_wiretap, text="3. Wiretap")
        self.notebook.add(self.frame_execution, text="4. Execution")
        self.notebook.add(self.frame_performance, text="5. Performance")
        self.notebook.add(self.frame_probing, text="6. Probing")
        self.notebook.add(self.frame_gallery, text="7. Error Gallery")
    def set_subject_loaded(self, name):
        self.lbl_subject.config(text=f"Subject: {name}", foreground="green")
    def set_evidence_loaded(self):
        self.lbl_evidence.config(text=f"Evidence: Mapped", foreground="green")
    def log(self, msg):
        print(f"[GUI LOG] {msg}")