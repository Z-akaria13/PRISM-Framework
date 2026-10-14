"""
GUI UTILITIES
Helper classes for the Tkinter frontend.
"""
import tkinter as tk
from tkinter import filedialog, messagebox

from prism.core.cache import AuditCache, CacheError

def create_info_button(parent, title, message):
    """
    Creates a small [?] button that pops up an info dialog when clicked.
    """
    def show_info():
        messagebox.showinfo(title, message)

    # We use a standard Tkinter button (not ttk) because it's easier
    # to make it tiny and square without theme interference.
    btn = tk.Button(
        parent,
        text="?",
        command=show_info,
        font=("Arial", 8, "bold"),
        fg="#0055cc",        # Blue text
        cursor="hand2",      # Pointer finger on hover
        relief="flat",
        padx=4, pady=0
    )
    return btn


def open_cache(app):
    """
    Opens the results cache for an analysis tab: the last extraction's output
    directory, or one the user picks. Problems are shown in a dialog; returns None then.
    """
    out_dir = app.active_cache_dir
    if not out_dir:
        messagebox.showinfo("Locate Cache", "Please select the folder holding the extraction's results "
                                            "(prism_embeddings.h5 and prism_metadata.csv).")
        out_dir = filedialog.askdirectory()
        if not out_dir:
            return None
    try:
        cache = AuditCache.open(out_dir)
    except CacheError as e:
        messagebox.showerror("Error", str(e))
        app.active_cache_dir = None
        return None
    app.active_cache_dir = out_dir # Save for the other tabs
    return cache


def column_mapping(app, cache):
    """
    (id_col, target_col, group_col) to analyse `cache` with: the Evidence tab's mapping if it
    was confirmed in this session, otherwise the mapping recorded when the cache was extracted.
    """
    if app.metadata_df is not None or cache.config is None:
        return app.config_id_col, app.config_target_col, app.config_group_col
    config = cache.config
    return config.id_col, config.target_col, config.group_col
