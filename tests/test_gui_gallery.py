"""Error Gallery tab: error listing, filters, suspect rendering and nearest neighbours."""
import h5py
import numpy as np
import pandas as pd
import pytest

from _support import ValueConnector, known_bug, run_engine, value_frame, wait_idle, write_legacy_cache
from prism.inspectors.sklearn import SklearnInspector

pytestmark = pytest.mark.gui

# Same example as the Performance tests: item 3 is a false negative, item 4 a false positive.
PROBS = [0.9, 0.2, 0.8, 0.4, 0.7, 0.1]
LABELS = [1, 0, 1, 1, 0, 0]
EMBEDDING = [[0.0], [10.0], [20.0], [30.0], [31.0], [32.0]]


class RecordingConnector:
    project_root = ""

    def __init__(self, render_type="text"):
        self.render_type = render_type
        self.row_ids = []

    def load_item(self, row_id, meta):
        self.row_ids.append(row_id)
        return meta["subject_id"]

    def render_item_for_gallery(self, item):
        if self.render_type == "table":
            return {"type": "table", "payload": {"Subject": item, "Age": 71}}
        return {"type": "text", "payload": f"scan of {item}"}


def gallery_cache(tmp_path, labels=LABELS, probs=PROBS):
    metadata = pd.DataFrame({"subject_id": [f"S{i:03d}" for i in range(len(labels))], "label": labels})
    return write_legacy_cache(tmp_path / "cache", metadata, probs, {"emb": EMBEDDING[:len(labels)]})


def open_tab(app, cache_dir, connector, target="label"):
    app.connector_instance = connector
    app.active_cache_dir = str(cache_dir)
    app.config_target_col = target
    app.config_id_col = "subject_id"
    tab = app.frame_gallery
    tab.load_data()
    return tab


def listed(tab):
    return [tuple(str(v) for v in tab.tree.item(i, "values")) for i in tab.tree.get_children()]


def test_every_item_is_listed_with_its_status(app, tmp_path):
    tab = open_tab(app, gallery_cache(tmp_path), RecordingConnector())

    rows = listed(tab)

    assert [r[0] for r in rows] == [f"S{i:03d}" for i in range(6)]
    assert [r[3] for r in rows] == ["Correct", "Correct", "Correct", "False Neg", "False Pos", "Correct"]
    assert list(tab.cb_layer["values"]) == ["emb"]


@pytest.mark.parametrize("choice, expected", [
    ("Errors Only", ["S003", "S004"]),
    ("False Positives", ["S004"]),
    ("False Negatives", ["S003"]),
])
def test_filters(app, tmp_path, choice, expected):
    tab = open_tab(app, gallery_cache(tmp_path), RecordingConnector())
    tab.cb_filter.set(choice)

    tab.populate_list()

    assert [r[0] for r in listed(tab)] == expected


def test_inspecting_an_item_renders_it_and_its_three_nearest_neighbours(app, tmp_path):
    connector = RecordingConnector()
    tab = open_tab(app, gallery_cache(tmp_path), connector)

    tab._render_suspect(3)
    tab._render_neighbors(3)

    assert len(tab.frm_suspect.winfo_children()) == 1
    assert len(tab.frm_lineup.winfo_children()) == 3
    assert connector.row_ids == ["S003", "S004", "S005", "S002"]  # suspect, then neighbours by distance


def test_results_without_layers_say_why_there_are_no_neighbours(app, tmp_path):
    metadata = pd.DataFrame({"subject_id": [f"S{i:03d}" for i in range(6)], "label": LABELS})
    cache = write_legacy_cache(tmp_path / "black_box", metadata, PROBS)  # e.g. a scikit-learn model
    tab = open_tab(app, cache, RecordingConnector())

    tab.inspect(3)
    wait_idle(app, tab.runner)

    assert len(tab.frm_suspect.winfo_children()) == 1
    note, = tab.frm_lineup.winfo_children()
    assert note.cget("text") == "These results have no layers to find neighbours in."


def test_selecting_a_row_in_the_list_triggers_inspection(app, tmp_path):
    connector = RecordingConnector()
    tab = open_tab(app, gallery_cache(tmp_path), connector)
    tab.cb_filter.set("False Positives")
    tab.populate_list()

    tab.tree.selection_set(tab.tree.get_children()[0])
    tab.on_select_item(None)
    wait_idle(app, tab.runner)

    assert connector.row_ids[0] == "S004"


def test_gallery_requires_a_loaded_connector(app, dialogs):
    app.frame_gallery.load_data()

    assert "No Connector Loaded" in dialogs.messages("showerror")[0]


def test_regression_targets_are_refused(app, tmp_path, dialogs):
    metadata = pd.DataFrame({"subject_id": [f"S{i}" for i in range(6)], "age": np.linspace(60, 80, 6)})
    cache = write_legacy_cache(tmp_path / "cache", metadata, np.zeros(6))

    open_tab(app, cache, RecordingConnector(), target="age")

    assert "Regression" in dialogs.messages("showinfo")[0]


def test_multiclass_targets_are_refused(app, tmp_path, dialogs):
    metadata = pd.DataFrame({"subject_id": [f"S{i}" for i in range(6)], "label": list("abcabc")})
    cache = write_legacy_cache(tmp_path / "cache", metadata, np.eye(3)[[0, 1, 2] * 2])

    open_tab(app, cache, RecordingConnector())

    assert "Binary" in dialogs.messages("showinfo")[0]


def test_table_payloads_are_rendered(app, tmp_path):
    tab = open_tab(app, gallery_cache(tmp_path), RecordingConnector(render_type="table"))

    tab._render_suspect(0)
    tab._render_neighbors(0)

    table = tab.frm_suspect.winfo_children()[0]
    assert [table.item(row, "values") for row in table.get_children()] == [("Subject", "S000"), ("Age", "71")]
    assert len(tab.frm_lineup.winfo_children()) == 3


def test_predictions_are_expressed_in_the_ground_truth_labels(app, tmp_path):
    tab = open_tab(app, gallery_cache(tmp_path, labels=["AD", "CN", "AD", "AD", "CN", "CN"]),
                   RecordingConnector())

    assert set(tab.y_pred) <= set(tab.y_true)


def test_the_gallery_loads_the_same_item_that_was_embedded(app, tmp_path):
    connector = ValueConnector(fail_load={1})
    run = run_engine(tmp_path / "out", value_frame(6), connector, SklearnInspector(None), batch_size=6,
                     max_failed_percent=100)
    extraction_id = dict((value, row_id) for row_id, value in connector.calls)
    connector.calls.clear()
    app.connector_instance = connector
    app.active_cache_dir = str(run.out_dir)
    app.config_target_col, app.config_id_col = "label", "subject_id"
    tab = app.frame_gallery
    tab.load_data()

    tab._render_suspect(1)  # cache row 1 is the original row with value 2

    gallery_row_id, value = connector.calls[0]
    assert value == 2
    assert gallery_row_id == extraction_id[2]


def test_a_new_session_uses_the_mapping_recorded_with_the_cache(app, tmp_path):
    connector = ValueConnector()
    run = run_engine(tmp_path / "out", value_frame(4), connector, SklearnInspector(None))
    app.connector_instance, app.active_cache_dir = connector, str(run.out_dir)  # nothing confirmed in Evidence
    connector.calls.clear()

    tab = app.frame_gallery
    tab.load_data()
    tab._render_suspect(2)

    assert [r[0] for r in listed(tab)] == ["S000", "S001", "S002", "S003"]
    assert connector.seen_row_ids == ["S002"]


def test_neighbours_come_from_other_groups(app, tmp_path):
    metadata = pd.DataFrame({"subject_id": [f"S{i:03d}" for i in range(6)], "label": LABELS,
                             "patient": ["P1", "P1", "P2", "P2", "P2", "P3"]})
    cache = write_legacy_cache(tmp_path / "cache", metadata, PROBS, {"emb": EMBEDDING})
    app.config_group_col = "patient"
    connector = RecordingConnector()
    tab = open_tab(app, cache, connector)

    tab._render_neighbors(3)  # S003 belongs to P2, as do its two closest points S004 and S002

    assert connector.row_ids == ["S005", "S001", "S000"]


def test_a_suspect_with_no_other_groups_gets_an_empty_lineup(app, tmp_path):
    metadata = pd.DataFrame({"subject_id": [f"S{i:03d}" for i in range(6)], "label": LABELS, "patient": "P1"})
    cache = write_legacy_cache(tmp_path / "cache", metadata, PROBS, {"emb": EMBEDDING})
    app.config_group_col = "patient"
    connector = RecordingConnector()
    tab = open_tab(app, cache, connector)

    tab._render_neighbors(0)

    assert connector.row_ids == []
    assert tab.frm_lineup.winfo_children()[0].cget("text") == "No other groups to compare with."



def test_the_search_space_layer_is_loaded_once_for_many_clicks(app, tmp_path, monkeypatch):
    tab = open_tab(app, gallery_cache(tmp_path), RecordingConnector())
    loads = []
    real_layer = tab.cache.layer
    monkeypatch.setattr(tab.cache, "layer", lambda name: loads.append(name) or real_layer(name))

    for suspect in (0, 3, 5, 3):
        tab._render_neighbors(suspect)

    assert loads == ["emb"]


def test_loading_new_results_refreshes_the_search_space(app, tmp_path):
    connector = RecordingConnector()
    tab = open_tab(app, gallery_cache(tmp_path / "first"), connector)
    tab._render_neighbors(3)
    other = gallery_cache(tmp_path / "second")
    with h5py.File(other / "prism_embeddings.h5", "a") as f:
        f["emb"][...] = np.array(EMBEDDING)[::-1]  # same rows, reversed distances
    app.active_cache_dir = str(other)
    connector.row_ids.clear()

    tab.load_data()
    tab._render_neighbors(3)

    # Suspect S003 now sits at 20: S002 and S004 are both 10 away (tie broken by row), then S001.
    # The stale layer would have given S004, S005, S002.
    assert connector.row_ids == ["S002", "S004", "S001"]



def test_inspecting_loads_items_off_the_tk_thread(app, tmp_path):
    import threading

    class ThreadRecorder(RecordingConnector):
        def __init__(self):
            super().__init__()
            self.on_tk_thread = []

        def load_item(self, row_id, meta):
            self.on_tk_thread.append(threading.current_thread() is threading.main_thread())
            return super().load_item(row_id, meta)

    connector = ThreadRecorder()
    tab = open_tab(app, gallery_cache(tmp_path), connector)

    tab.inspect(3)
    assert tab.frm_suspect.winfo_children()[0].cget("text") == "Loading..."
    wait_idle(app, tab.runner)

    assert connector.on_tk_thread == [False] * 4
    assert connector.row_ids == ["S003", "S004", "S005", "S002"]
    assert len(tab.frm_lineup.winfo_children()) == 3


def test_a_connector_error_while_inspecting_is_shown(app, tmp_path, dialogs):
    class Broken(RecordingConnector):
        def load_item(self, row_id, meta):
            raise IOError("scan file missing")

    tab = open_tab(app, gallery_cache(tmp_path), Broken())

    tab.inspect(0)
    wait_idle(app, tab.runner)

    assert "scan file missing" in dialogs.messages("showerror")[0]


def test_clicking_quickly_shows_only_the_last_item(app, tmp_path):
    connector = RecordingConnector()
    tab = open_tab(app, gallery_cache(tmp_path), connector)

    for idx in (0, 1, 2, 3):
        tab.inspect(idx)
    wait_idle(app, tab.runner)

    shown = tab.frm_suspect.winfo_children()[0].get("1.0", "end").strip()
    assert shown == "scan of S003"
