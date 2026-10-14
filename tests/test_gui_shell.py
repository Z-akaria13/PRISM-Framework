"""Main window, Setup tab, Wiretap tab and Execution tab guards."""
from types import SimpleNamespace

import pytest

from _support import BLACK_BOX_CONNECTOR, TORCH_CONNECTOR, ingest, make_project, wait_for, write_script

pytestmark = pytest.mark.gui


def tree_texts(tree):
    return [tree.item(i, "text") for i in tree.get_children()]


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

def test_app_builds_the_seven_tabs_in_order(app):
    tabs = [app.notebook.tab(tab, "text") for tab in app.notebook.tabs()]

    assert tabs == ["1. Setup", "2. Evidence", "3. Wiretap", "4. Execution",
                    "5. Performance", "6. Probing", "7. Error Gallery"]


def test_window_title_shows_the_installed_version(app):
    import prism

    assert app.title() == f"PRISM // v{prism.__version__}"


def test_app_starts_with_nothing_loaded(app):
    assert app.connector_instance is None
    assert app.model_subject is None
    assert app.inspector is None
    assert app.metadata_df is None
    assert (app.config_id_col, app.config_target_col, app.config_group_col) == (None, None, None)
    assert app.active_cache_dir is None
    wait_for(app, app._device_task)  # checked in the background
    assert app.lbl_device.cget("text") == "Running on CPU"


# ---------------------------------------------------------------------------
# Setup tab
# ---------------------------------------------------------------------------

def test_framework_picker_lists_every_plugin(app):
    assert set(app.frame_setup.cb_framework["values"]) == {"PyTorch", "TensorFlow/Keras", "Scikit-Learn"}


def test_ingesting_a_pytorch_connector_populates_the_wiretap(app, tmp_path):
    pytest.importorskip("torch")
    from prism.inspectors.pytorch import PyTorchInspector
    script = write_script(tmp_path / "connector.py", TORCH_CONNECTOR)

    ingest(app, tmp_path, script, "PyTorch")

    assert isinstance(app.inspector, PyTorchInspector)
    assert app.model_subject is not None
    assert (app.project_root, app.script_path, app.framework) == (str(tmp_path), str(script), "PyTorch")
    assert tree_texts(app.frame_wiretap.tree) == ["conv", "pool", "fc"]
    assert app.lbl_subject.cget("text") == "Subject: Loaded"


def test_ingesting_a_black_box_connector_empties_the_wiretap(app, tmp_path):
    app.frame_wiretap.populate_tree([{"id": "stale", "type": "Linear", "params": "1"}])
    script = write_script(tmp_path / "connector.py", BLACK_BOX_CONNECTOR)

    ingest(app, tmp_path, script, "Scikit-Learn")

    assert app.model_subject == "black-box-model"
    assert tree_texts(app.frame_wiretap.tree) == []


def test_ingest_without_a_script_says_so(app, dialogs, monkeypatch):
    started = []
    monkeypatch.setattr(app.frame_setup, "ingest", lambda *args: started.append(args))

    app.frame_setup.run_ingest()

    assert started == []
    assert dialogs.calls == [("showinfo", "No Connector Script", "Choose a connector script first.")]


def test_a_failed_ingest_is_reported_to_the_user(app, tmp_path, dialogs):
    ingest(app, tmp_path, tmp_path / "missing.py", "PyTorch")

    assert app.connector_instance is None
    assert "Ingestion failed" in [title for kind, title, _ in dialogs.calls if kind == "showerror"]
    assert app.frame_setup.lbl_status.cget("text") == "Ingestion failed."
    assert str(app.frame_setup.btn_ingest.cget("state")) == "normal"


# ---------------------------------------------------------------------------
# Wiretap tab
# ---------------------------------------------------------------------------

LAYERS = [
    {"id": "conv", "type": "Conv2d", "params": "30"},
    {"id": "layer1.0.relu", "type": "ReLU", "params": "0"},
    {"id": "fc", "type": "Linear", "params": "8"},
]


def test_populate_tree_replaces_previous_layers(app):
    wiretap = app.frame_wiretap
    wiretap.populate_tree([{"id": "old", "type": "Linear", "params": "1"}])

    wiretap.populate_tree(LAYERS)

    assert tree_texts(wiretap.tree) == ["conv", "layer1.0.relu", "fc"]
    first = wiretap.tree.get_children()[0]
    assert [str(v) for v in wiretap.tree.item(first, "values")] == ["☐", "Conv2d", "30", "Mean Pool", "", ""]
    assert wiretap.extracted_layers() == {}  # nothing is extracted until ticked


def test_ticked_layers_are_extracted_with_their_pooling_in_model_order(app):
    wiretap = app.frame_wiretap
    wiretap.populate_tree(LAYERS)
    conv, relu, fc = wiretap.tree.get_children()
    wiretap.tree.selection_set((fc, conv))

    wiretap.set_ticked(True)
    wiretap.tree.selection_set(relu)  # selecting a row doesn't add it

    assert list(wiretap.extracted_layers().items()) == [("conv", "Mean Pool"), ("fc", "Mean Pool")]


def test_set_strategy_changes_the_selected_rows_only(app):
    wiretap = app.frame_wiretap
    wiretap.populate_tree(LAYERS)
    items = wiretap.tree.get_children()
    wiretap.tree.selection_set(items)
    wiretap.set_ticked(True)
    wiretap.tree.selection_set(items[:2])

    wiretap.set_strategy("Max Pool")
    wiretap.set_strategy("Flatten")

    assert tree_texts(wiretap.tree) == ["conv", "layer1.0.relu", "fc"]  # names stay clean
    assert wiretap.extracted_layers() == {"conv": "Flatten", "layer1.0.relu": "Flatten", "fc": "Mean Pool"}


def right_click(wiretap, monkeypatch, item):
    monkeypatch.setattr(wiretap.tree, "identify_row", lambda y: item)
    monkeypatch.setattr(wiretap.context_menu, "post", lambda x, y: None)
    wiretap.show_context_menu(SimpleNamespace(x=0, y=0, x_root=0, y_root=0))


def test_each_layer_can_get_its_own_pooling_without_losing_the_ticks(app, monkeypatch):
    wiretap = app.frame_wiretap
    wiretap.populate_tree(LAYERS)
    conv, relu, fc = wiretap.tree.get_children()
    wiretap.tree.selection_set((conv, relu, fc))
    wiretap.set_ticked(True)

    wiretap.tree.selection_set(fc)
    right_click(wiretap, monkeypatch, relu)  # outside the selection: selects just relu
    wiretap.set_strategy("Seq Pool")
    right_click(wiretap, monkeypatch, conv)
    wiretap.set_strategy("Grid Pool")

    assert wiretap.extracted_layers() == {"conv": "Grid Pool", "layer1.0.relu": "Seq Pool", "fc": "Mean Pool"}


def click_box(wiretap, monkeypatch, item, column="#1", region="cell"):
    monkeypatch.setattr(wiretap.tree, "identify_region", lambda x, y: region)
    monkeypatch.setattr(wiretap.tree, "identify_column", lambda x: column)
    monkeypatch.setattr(wiretap.tree, "identify_row", lambda y: item)
    return wiretap.on_click(SimpleNamespace(x=0, y=0))


def test_clicking_a_box_ticks_that_row_and_keeps_the_selection(app, monkeypatch):
    wiretap = app.frame_wiretap
    wiretap.populate_tree(LAYERS)
    conv, relu, fc = wiretap.tree.get_children()
    wiretap.tree.selection_set((conv, relu))

    assert click_box(wiretap, monkeypatch, fc) == "break"  # Tk does not go on to select the row
    assert wiretap.extracted_layers() == {"fc": "Mean Pool"}
    click_box(wiretap, monkeypatch, fc)
    assert wiretap.extracted_layers() == {}
    assert click_box(wiretap, monkeypatch, fc, column="#4") is None  # elsewhere in the row: an ordinary click
    assert wiretap.extracted_layers() == {}
    assert set(wiretap.tree.selection()) == {conv, relu}


def test_space_ticks_the_selection_or_unticks_it_when_all_are_ticked(app):
    wiretap = app.frame_wiretap
    wiretap.populate_tree(LAYERS)
    conv, relu, fc = wiretap.tree.get_children()
    wiretap.tree.selection_set(conv)
    wiretap.set_ticked(True)
    wiretap.tree.selection_set((conv, relu))

    wiretap.toggle_selected()
    assert set(wiretap.extracted_layers()) == {"conv", "layer1.0.relu"}
    wiretap.toggle_selected()
    assert wiretap.extracted_layers() == {}


def torch_subject_with_metadata(app, tmp_path):
    """A PyTorch subject ingested through Setup, and its metadata confirmed in Evidence."""
    pytest.importorskip("torch")
    script, csv = make_project(tmp_path / "project", TORCH_CONNECTOR, n=4)
    ingest(app, script.parent, script, "PyTorch")
    evidence = app.frame_evidence
    evidence.load_csv(str(csv))
    evidence.cb_id.set("subject_id")
    evidence.cb_target.set("label")
    evidence.confirm_mapping()
    return app.frame_wiretap


def test_show_shapes_fills_each_layers_output_and_what_its_pooling_keeps(app, tmp_path):
    wiretap = torch_subject_with_metadata(app, tmp_path)

    wait_for(app, wiretap.show_shapes())

    rows = {wiretap.tree.item(i, "text"): (wiretap.tree.set(i, "Output"), wiretap.tree.set(i, "Stored"))
            for i in wiretap.tree.get_children()}
    assert rows == {"conv": ("3×4×4", "3"), "pool": ("3×1×1", "3"), "fc": ("2", "2")}
    assert wiretap.lbl_status.cget("text") == "Shapes from row S000."
    assert not app.model_subject.training  # the connector's eval mode is kept
    assert not app.model_subject.conv._forward_hooks  # and no hook is left behind

    conv = wiretap.tree.get_children()[0]
    wiretap.tree.selection_set(conv)
    wiretap.set_strategy("Flatten")
    assert wiretap.tree.set(conv, "Stored") == "48"
    wiretap.set_strategy("Seq Pool")
    assert wiretap.tree.set(conv, "Stored") == "16 (4×4)"
    wiretap.set_strategy("Grid Pool")
    assert wiretap.tree.set(conv, "Stored") == "48"


def test_show_shapes_says_what_it_needs(app, tmp_path, dialogs):
    wiretap = app.frame_wiretap

    wiretap.show_shapes()  # nothing ingested
    script = write_script(tmp_path / "connector.py", BLACK_BOX_CONNECTOR)
    ingest(app, tmp_path, script, "Scikit-Learn")
    wiretap.show_shapes()  # a black box: no layers
    wiretap.populate_tree(LAYERS)
    wiretap.show_shapes()  # layers, but no metadata confirmed

    assert [(kind, title) for kind, title, _ in dialogs.calls] == [
        ("showerror", "No Subject Loaded"), ("showinfo", "No Layers"), ("showinfo", "Metadata Needed")]
    assert wiretap.task is None


def test_a_failing_item_is_reported_by_show_shapes(app, tmp_path, dialogs, monkeypatch):
    wiretap = torch_subject_with_metadata(app, tmp_path)
    monkeypatch.setattr(app.connector_instance, "load_item", lambda row_id, meta: open("missing/scan.nii"))

    wait_for(app, wiretap.show_shapes())

    error, = dialogs.messages("showerror")
    assert error.startswith("Running row S000 through the model failed:") and "missing" in error
    assert wiretap.lbl_status.cget("text") == "Show shapes failed."
    assert str(wiretap.btn_shapes.cget("state")) == "normal"


def test_the_execution_tab_extracts_the_ticked_layers_with_their_pooling(app, tmp_path):
    wiretap = torch_subject_with_metadata(app, tmp_path)
    conv, pool, fc = wiretap.tree.get_children()
    wiretap.tree.selection_set((conv, fc))
    wiretap.set_ticked(True)
    wiretap.tree.selection_set(conv)
    wiretap.set_strategy("Max Pool")

    config = app.frame_execution.build_config(str(tmp_path / "out"), 4)

    assert config.layers == {"conv": "Max Pool", "fc": "Mean Pool"}


def test_right_click_keeps_the_multi_selection(app, monkeypatch):
    wiretap = app.frame_wiretap
    wiretap.populate_tree(LAYERS)
    items = wiretap.tree.get_children()
    wiretap.tree.selection_set(items)

    right_click(wiretap, monkeypatch, items[1])
    wiretap.set_strategy("Max Pool")

    assert set(wiretap.tree.selection()) == set(items)
    assert {wiretap.tree.set(item, "Pooling") for item in items} == {"Max Pool"}


# ---------------------------------------------------------------------------
# Execution tab guards
# ---------------------------------------------------------------------------

def test_execution_requires_a_loaded_subject(app, dialogs):
    app.frame_execution.start_execution()

    assert dialogs.calls[0][0] == "showerror"
    assert "No Subject Loaded" in dialogs.calls[0][2]


def test_execution_requires_confirmed_evidence(app, dialogs):
    app.model_subject = object()

    app.frame_execution.start_execution()

    assert dialogs.calls and "No Evidence Loaded" in dialogs.calls[0][2]


def test_execution_requires_an_output_directory(app, dialogs):
    app.model_subject = object()
    app.metadata_df = object()

    app.frame_execution.start_execution()

    assert "output directory" in dialogs.messages("showerror")[0]


def test_execution_rejects_an_invalid_batch_size(app, dialogs, tmp_path):
    execution = app.frame_execution
    app.model_subject, app.metadata_df = object(), object()
    execution.ent_outdir.insert(0, str(tmp_path))
    execution.ent_batch.delete(0, "end")
    execution.ent_batch.insert(0, "sixteen")

    execution.start_execution()

    assert "Invalid execution settings" in dialogs.messages("showerror")[0]
    assert str(execution.btn_run.cget("state")) != "disabled"


def test_device_defaults_to_auto(app):
    assert app.frame_execution.cb_device.get() == "auto"
    assert tuple(app.frame_execution.cb_device["values"]) == ("auto", "cpu", "cuda")


def test_grid_pool_can_be_chosen_from_the_right_click_menu(app):
    wiretap = app.frame_wiretap
    wiretap.populate_tree(LAYERS)
    labels = [wiretap.context_menu.entrycget(i, "label") for i in range(wiretap.context_menu.index("end") + 1)
              if wiretap.context_menu.type(i) == "command"]
    grid = next(i for i in range(wiretap.context_menu.index("end") + 1)
                if wiretap.context_menu.type(i) == "command" and "Grid Pool" in wiretap.context_menu.entrycget(i, "label"))
    wiretap.tree.selection_set(wiretap.tree.get_children()[0])

    wiretap.context_menu.invoke(grid)

    assert any("Grid Pool" in label for label in labels)
    assert wiretap.tree.set(wiretap.tree.get_children()[0], "Pooling") == "Grid Pool"



def test_ingestion_runs_in_the_background(app, tmp_path):
    import threading
    script = write_script(tmp_path / "connector.py", BLACK_BOX_CONNECTOR)
    started, release = threading.Event(), threading.Event()
    import prism.core.subject as subject_module
    import prism.gui.frames.setup as setup_module
    real_load = subject_module.load_subject

    def slow_load(*args, **kwargs):
        started.set()
        release.wait(5)
        return real_load(*args, **kwargs)

    setup_module.load_subject, saved = slow_load, setup_module.load_subject
    try:
        task = app.frame_setup.ingest(str(tmp_path), str(script), "Scikit-Learn")
        assert started.wait(5)
        app.update()  # the window keeps responding while the model loads
        assert str(app.frame_setup.btn_ingest.cget("state")) == "disabled"
        assert app.frame_setup.lbl_status.cget("text") == "Loading connector and model..."
        release.set()
        wait_for(app, task)
    finally:
        setup_module.load_subject = saved

    assert app.model_subject == "black-box-model"
    assert str(app.frame_setup.btn_ingest.cget("state")) == "normal"


def test_right_click_outside_the_selection_selects_that_row(app, monkeypatch):
    wiretap = app.frame_wiretap
    wiretap.populate_tree(LAYERS)
    items = wiretap.tree.get_children()
    wiretap.tree.selection_set(items[:2])
    monkeypatch.setattr(wiretap.tree, "identify_row", lambda y: items[2])
    monkeypatch.setattr(wiretap.context_menu, "post", lambda x, y: None)

    wiretap.show_context_menu(SimpleNamespace(x=0, y=0, x_root=0, y_root=0))

    assert wiretap.tree.selection() == (items[2],)
