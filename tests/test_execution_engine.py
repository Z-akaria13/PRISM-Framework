"""core.execution.ExecutionEngine: batching, fault tolerance, pooling and the HDF5 cache."""
import h5py
import numpy as np
import pytest

from _support import (H5_NAME, TensorConnector, ValueConnector, capture_activation,
                      known_bug, run_engine, tiny_cnn, value_frame)
from prism.core.cache import AuditCache
from prism.inspectors.sklearn import SklearnInspector

torch = pytest.importorskip("torch")
nn = torch.nn
from prism.inspectors.pytorch import PyTorchInspector  # noqa: E402  (needs torch)


def black_box(tmp_path, metadata, connector, **kwargs):
    return run_engine(tmp_path / "out", metadata, connector, SklearnInspector(None), **kwargs)


def white_box(tmp_path, metadata, connector, model, layers, **kwargs):
    return run_engine(tmp_path / "out", metadata, connector, PyTorchInspector(model),
                      model=model, layers=layers, **kwargs)


def assert_aligned(run):
    """Every dataset in the cache must have exactly one row per row of the metadata CSV."""
    n_rows = len(run.metadata)
    lengths = {name: len(values) for name, values in run.datasets().items()}
    assert lengths and all(n == n_rows for n in lengths.values()), (lengths, n_rows)
    assert run.ids() == list(run.metadata["subject_id"])
    AuditCache.open(str(run.out_dir))  # and the reader accepts it


# ---------------------------------------------------------------------------
# Black-box path, batching and fault tolerance
# ---------------------------------------------------------------------------

def test_black_box_run_writes_outputs_and_metadata(tmp_path):
    metadata = value_frame(10)

    run = black_box(tmp_path, metadata, ValueConnector(), batch_size=4)

    assert run.ok
    datasets = run.datasets()
    assert set(datasets) == {"__model_output__"}
    np.testing.assert_array_equal(datasets["__model_output__"], np.arange(10))
    assert run.metadata.equals(metadata)
    assert any("[SUCCESS] Extracted 10/10" in line for line in run.logs)


def test_progress_is_reported_per_batch_and_ends_at_100(tmp_path):
    run = black_box(tmp_path, value_frame(10), ValueConnector(), batch_size=4)

    assert run.progress == pytest.approx([100 / 3, 200 / 3, 100.0])


def test_rows_that_fail_to_load_are_dropped_and_outputs_stay_aligned(tmp_path):
    run = black_box(tmp_path, value_frame(10), ValueConnector(fail_load={3, 7}), batch_size=4,
                    max_failed_percent=100)

    assert run.ok
    assert list(run.metadata["value"]) == [0, 1, 2, 4, 5, 6, 8, 9]
    np.testing.assert_array_equal(run.datasets()["__model_output__"], run.metadata["value"])
    assert sum("Skipping row" in line for line in run.logs) == 2
    failed = run.failed_rows
    assert list(failed["subject_id"]) == ["S003", "S007"]
    assert list(failed["prism_failed_at"]) == ["load_item", "load_item"]
    assert list(failed["prism_error"]) == ["OSError: cannot read item 3", "OSError: cannot read item 7"]


def test_batch_whose_inference_crashes_is_dropped_whole(tmp_path):
    run = black_box(tmp_path, value_frame(8), ValueConnector(fail_infer={5}), batch_size=4, max_failed_percent=100)

    assert run.ok
    assert list(run.metadata["value"]) == [0, 1, 2, 3]
    np.testing.assert_array_equal(run.datasets()["__model_output__"], [0, 1, 2, 3])
    assert any("Batch 1 inference crashed" in line for line in run.logs)
    assert list(run.failed_rows["value"]) == [4, 5, 6, 7]
    assert set(run.failed_rows["prism_failed_at"]) == {"run_inference"}
    assert set(run.failed_rows["prism_error"]) == {"RuntimeError: inference blew up"}


def test_batch_where_every_row_fails_to_load_is_skipped(tmp_path):
    run = black_box(tmp_path, value_frame(6), ValueConnector(fail_load={2, 3}), batch_size=2, max_failed_percent=100)

    assert list(run.metadata["value"]) == [0, 1, 4, 5]
    assert any("Batch 1 is empty" in line for line in run.logs)
    assert list(run.failed_rows["value"]) == [2, 3]


def test_run_inference_returning_none_drops_the_batch(tmp_path):
    class ReturnsNone(ValueConnector):
        def run_inference(self, model, batch):
            return None if 2 in batch else super().run_inference(model, batch)

    run = black_box(tmp_path, value_frame(4), ReturnsNone(), batch_size=2, max_failed_percent=100)

    assert list(run.metadata["value"]) == [0, 1]
    assert sum("inference crashed" in line and "returned None" in line for line in run.logs) == 1
    assert set(run.failed_rows["prism_error"]) == {"ValueError: run_inference() returned None."}


def test_a_run_without_failures_writes_no_list_of_them(tmp_path):
    run = black_box(tmp_path, value_frame(4), ValueConnector())

    assert run.ok and run.failed_rows is None
    assert not any(line.startswith("[WARNING]") for line in run.logs)


# ---------------------------------------------------------------------------
# Failed rows: recorded, described, limited (debt P0 #22, P1 #23)
# ---------------------------------------------------------------------------

def sites_frame(n=100, unreadable_site="B", unreadable=30):
    """`n` rows over sites A and B; the last `unreadable` rows are site B's."""
    frame = value_frame(n)
    frame["site"] = ["A"] * (n - unreadable) + [unreadable_site] * unreadable
    frame["sex"] = ["F", "M"] * (n // 2)
    return frame


def test_a_site_whose_files_fail_is_reported_not_silently_dropped(tmp_path):
    # The debt.MD case: 100 rows, site B's 30 files unreadable.
    metadata = sites_frame()

    run = black_box(tmp_path, metadata, ValueConnector(fail_load=set(range(70, 100))), max_failed_percent=100)

    assert run.ok and len(run.metadata) == 70
    assert len(run.failed_rows) == 30 and set(run.failed_rows["site"]) == {"B"}
    warnings = run.logs[-4:-1]  # just before [SUCCESS]
    assert warnings[0] == ("[WARNING] 30 of 100 rows (30%) failed and are left out of the results: 30 in load_item "
                           "(first error: OSError: cannot read item 70).")
    assert warnings[1] == "[WARNING] Most affected: site = B (all 30 rows: missing from every analysis)."
    assert "prism_failed_rows.csv lists each one" in warnings[2]

    cache = AuditCache.open(str(run.out_dir))
    assert (cache.rows_attempted, cache.rows_failed) == (100, 30)
    assert cache.warnings[0] == cache.failure_note
    assert cache.failure_note.startswith("30 of 100 rows (30%) failed during the extraction and are not in these "
                                         "results: 30 in load_item")
    assert "Most affected: site = B (all 30 rows: missing from every analysis)." in cache.failure_note
    assert list(cache.failures["subject_id"]) == [f"S{i:03d}" for i in range(70, 100)]


def test_failures_spread_evenly_name_no_group(tmp_path):
    # 3 of site A's 70 rows and 2 of site B's 30; 3 of each label's 50 rows and 2 of the other's.
    run = black_box(tmp_path, sites_frame(), ValueConnector(fail_load={10, 31, 52, 73, 94}), max_failed_percent=100)

    assert run.ok
    assert not any("Most affected" in line for line in run.logs)
    assert "Most affected" not in AuditCache.open(str(run.out_dir)).failure_note


def test_the_id_and_group_by_columns_are_not_reported_as_groups(tmp_path):
    metadata = sites_frame(n=12, unreadable=4)
    metadata["patient"] = [i // 4 for i in range(12)]  # the failed rows are all of patient 2's scans

    run = black_box(tmp_path, metadata, ValueConnector(fail_load={8, 9, 10, 11}), group_col="patient",
                    max_failed_percent=100)

    groups = next(line for line in run.logs if "Most affected" in line)
    assert "site = B" in groups and "patient" not in groups and "subject_id" not in groups


def test_the_run_stops_once_more_rows_failed_than_the_limit_allows(tmp_path):
    connector = ValueConnector(fail_load=set(range(70, 100)))

    run = black_box(tmp_path, sites_frame(), connector, batch_size=10)  # the default limit: 5%

    assert not run.ok
    assert len(connector.calls) == 76  # stopped at the 6th failure, the first above 5 of 100
    assert run.logs[-3] == ("[WARNING] 6 of the 76 rows tried so far failed: 6 in load_item "
                            "(first error: OSError: cannot read item 70).")
    assert run.logs[-2] == "[WARNING] Most affected so far: site = B (all 6 rows: missing from every analysis)."
    assert run.logs[-1].startswith("[CRITICAL ERROR] Stopped: more than 5% of the 100 rows failed, so nothing was "
                                   "saved.")
    assert "max_failed_percent" in run.logs[-1]
    assert not any((run.out_dir / name).exists() for name in (H5_NAME, "prism_metadata.csv"))
    assert list(run.out_dir.iterdir()) == []  # no .partial files left behind either


@pytest.mark.parametrize("limit, failures, ok", [(5, 5, True), (5, 6, False), (0, 1, False), (100, 99, True)])
def test_the_limit_is_a_share_of_all_rows(tmp_path, limit, failures, ok):
    run = black_box(tmp_path, value_frame(100), ValueConnector(fail_load=set(range(failures))),
                    max_failed_percent=limit)

    assert run.ok == ok


def test_a_run_where_every_row_fails_is_not_a_success(tmp_path):
    run = black_box(tmp_path, value_frame(6), ValueConnector(fail_load=set(range(6))), max_failed_percent=100)

    assert not run.ok
    assert not any("[SUCCESS]" in line for line in run.logs)
    assert run.logs[-2] == "[WARNING] 6 of 6 rows failed: 6 in load_item (first error: OSError: cannot read item 0)."
    assert run.logs[-1] == ("[CRITICAL ERROR] No rows were extracted: all 6 rows failed, so there are no results "
                            "to analyse.")
    assert list(run.out_dir.iterdir()) == []


def test_a_run_whose_every_batch_crashes_is_not_a_success(tmp_path):
    run = black_box(tmp_path, value_frame(4), ValueConnector(fail_infer={0, 2}), batch_size=2, max_failed_percent=100)

    assert not run.ok
    assert run.logs[-1].startswith("[CRITICAL ERROR] No rows were extracted: all 4 rows failed")


def test_empty_metadata_is_not_a_success(tmp_path):
    run = black_box(tmp_path, value_frame(0), ValueConnector())

    assert not run.ok
    assert run.logs[-1] == ("[CRITICAL ERROR] No rows were extracted: the metadata has no rows, so there are no "
                            "results to analyse.")


def test_h5_file_is_released_after_a_crash(tmp_path):
    def exploding_progress(value):
        raise RuntimeError("progress callback failed")

    run = black_box(tmp_path, value_frame(4), ValueConnector(), progress_callback=exploding_progress)

    assert not run.ok
    assert any("[CRITICAL ERROR]" in line for line in run.logs)
    h5py.File(run.out_dir / H5_NAME, "w").close()  # would raise if the engine still held it open


# ---------------------------------------------------------------------------
# White-box path (PyTorch hooks + pooling strategies)
# ---------------------------------------------------------------------------

def test_white_box_run_without_selected_layers_fails_cleanly(tmp_path):
    model = tiny_cnn()

    run = white_box(tmp_path, value_frame(4), TensorConnector((1, 4, 4)), model, layers=())

    assert not run.ok
    assert any("No layers chosen" in line for line in run.logs)


def test_torch_outputs_are_stored_as_numpy(tmp_path):
    model, conn, metadata = tiny_cnn(), TensorConnector((1, 4, 4)), value_frame(6)
    with torch.no_grad():
        expected = model(conn.inputs_for(metadata)).numpy()

    run = white_box(tmp_path, metadata, conn, model, layers=["fc"])

    assert run.ok
    datasets = run.datasets()
    np.testing.assert_allclose(datasets["__model_output__"], expected, rtol=1e-5, atol=1e-6)
    assert datasets["fc"].shape == (6, 2)


def test_hooks_are_removed_after_the_run(tmp_path):
    model = tiny_cnn()

    white_box(tmp_path, value_frame(4), TensorConnector((1, 4, 4)), model, layers=["conv", "fc"])

    assert all(len(module._forward_hooks) == 0 for module in model.modules())


@pytest.mark.parametrize("layers, reduce", [
    (["conv"], lambda a: a.mean(axis=(2, 3))),  # no strategy chosen -> default
    ({"conv": "Mean Pool"}, lambda a: a.mean(axis=(2, 3))),
    ({"conv": "Max Pool"}, lambda a: a.max(axis=(2, 3))),
    ({"conv": "Flatten"}, lambda a: a.reshape(len(a), -1)),
])
def test_spatial_pooling_strategies(tmp_path, layers, reduce):
    model, conn, metadata = tiny_cnn(), TensorConnector((1, 4, 4)), value_frame(10)
    reference = capture_activation(model, "conv", conn.inputs_for(metadata))

    run = white_box(tmp_path, metadata, conn, model, layers=layers, batch_size=4)

    assert run.ok
    np.testing.assert_allclose(run.datasets()["conv"], reduce(reference), rtol=1e-5, atol=1e-6)


class TinySequenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.proj = nn.Linear(4, 5)
        self.head = nn.Linear(5, 2)

    def forward(self, x):
        return self.head(self.proj(x).mean(dim=1))


@pytest.mark.parametrize("layers, reduce", [
    ({"proj": "Seq Pool"}, lambda a: a.mean(axis=1)),
    ({"proj": "CLS Token"}, lambda a: a[:, 0, :]),
    # The default "Mean Pool" assumes (B, C, spatial...). On a (B, T, D) sequence it averages the
    # embedding axis and leaves one value per token, so transformer users must pick Seq Pool / CLS.
    (["proj"], lambda a: a.mean(axis=2)),
])
def test_sequence_pooling_strategies(tmp_path, layers, reduce):
    model, conn, metadata = TinySequenceModel().eval(), TensorConnector((6, 4)), value_frame(5)
    reference = capture_activation(model, "proj", conn.inputs_for(metadata))

    run = white_box(tmp_path, metadata, conn, model, layers=layers, batch_size=2)

    np.testing.assert_allclose(run.datasets()["proj"], reduce(reference), rtol=1e-5, atol=1e-6)


def test_nested_layer_names_round_trip(tmp_path):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Sequential(nn.Conv2d(1, 2, 3, padding=1)), nn.Flatten()).eval()

    run = white_box(tmp_path, value_frame(4), TensorConnector((1, 4, 4)), model,
                    layers={"0.0": "Max Pool"})

    assert run.datasets()["0.0"].shape == (4, 2)


class DeviceAwareInspector(SklearnInspector):
    def __init__(self):
        super().__init__(None)
        self.devices = []

    def to_device(self, device):
        self.devices.append(device)
        return "cuda:7" if device == "auto" else device


@pytest.mark.parametrize("requested, used", [("auto", "cuda:7"), ("cpu", "cpu")])
def test_the_configured_device_is_applied_when_the_run_starts(tmp_path, requested, used):
    inspector = DeviceAwareInspector()

    run = run_engine(tmp_path / "out", value_frame(3), ValueConnector(), inspector, device=requested)

    assert run.ok
    assert inspector.devices == [requested]
    assert f"Running on device: {used}" in run.logs


def test_inspectors_without_device_support_are_left_alone(tmp_path):
    run = black_box(tmp_path, value_frame(3), ValueConnector())

    assert not any("Running on device" in line for line in run.logs)


def test_white_box_run_moves_the_model_to_the_configured_device(tmp_path):
    model = tiny_cnn()
    moves = []
    original_to = model.to
    model.to = lambda device: moves.append(device) or original_to(device)

    run = white_box(tmp_path, value_frame(3), TensorConnector((1, 4, 4)), model, layers=["fc"], device="cpu")

    assert run.ok and moves == ["cpu"]


def test_per_item_inference_loop_is_reassembled(tmp_path):
    """The template tells users they may loop over batch_data; hooks then fire once per item."""
    model, metadata = tiny_cnn(), value_frame(10)
    conn = TensorConnector((1, 4, 4), per_item=True)
    reference = capture_activation(model, "conv", conn.inputs_for(metadata))

    run = white_box(tmp_path, metadata, conn, model, layers={"conv": "Mean Pool"}, batch_size=4)

    np.testing.assert_allclose(run.datasets()["conv"], reference.mean(axis=(2, 3)), rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Known bugs
# ---------------------------------------------------------------------------

class ReusedActivation(nn.Module):
    """One ReLU module applied twice per forward pass, as in torchvision's BasicBlock."""

    def __init__(self, conv=False):
        super().__init__()
        torch.manual_seed(0)
        self.body = nn.Conv2d(2, 2, 3, padding=1) if conv else nn.Linear(3, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.body(self.relu(x)))


def every_call(model, layer_name, inputs):
    """Reference activations of every call of one module in a single forward pass."""
    module = dict(model.named_modules())[layer_name]
    calls = []
    handle = module.register_forward_hook(lambda mod, args, out: calls.append(out.detach().numpy().copy()))
    try:
        with torch.no_grad():
            model(inputs)
    finally:
        handle.remove()
    return calls


@pytest.mark.parametrize("batch_size", [1, 3, 4])
def test_each_call_of_a_reused_module_is_stored_separately(tmp_path, batch_size):
    """A ResNet-style block applies one ReLU twice: before the second layer and after it."""
    model, conn, metadata = ReusedActivation().eval(), TensorConnector((3,)), value_frame(4)
    first_call, second_call = every_call(model, "relu", conn.inputs_for(metadata))

    run = run_engine(tmp_path / "out", metadata, conn, PyTorchInspector(model), model=model,
                     layers=["relu"], batch_size=batch_size)

    stored = run.datasets()
    assert run.ok and set(stored) == {"__model_output__", "relu@0", "relu@1"}
    np.testing.assert_allclose(stored["relu@0"], first_call, rtol=1e-6)
    np.testing.assert_allclose(stored["relu@1"], second_call, rtol=1e-6)
    assert any("'relu' runs 2 times in each forward pass" in line for line in run.logs)


def test_each_call_of_a_reused_module_gets_the_selected_pooling(tmp_path):
    model = ReusedActivation(conv=True).eval()

    run = white_box(tmp_path, value_frame(4), TensorConnector((2, 4, 4)), model,
                    layers={"relu": "Mean Pool"})

    stored = run.datasets()
    assert stored["relu@0"].shape == (4, 2) and stored["relu@1"].shape == (4, 2)
    with h5py.File(run.out_dir / H5_NAME, "r") as f:
        attrs = dict(f["relu@1"].attrs)
        assert list(attrs.pop("output_shape")) == [2, 4, 4]
        assert attrs == {"pooling": "Mean Pool", "module": "relu", "call": 1}


def test_a_reused_module_inside_a_per_item_loop_still_lines_up(tmp_path):
    model, metadata = ReusedActivation().eval(), value_frame(5)
    conn = TensorConnector((3,), per_item=True)
    first_call, second_call = every_call(model, "relu", conn.inputs_for(metadata))

    run = run_engine(tmp_path / "out", metadata, conn, PyTorchInspector(model), model=model,
                     layers=["relu"], batch_size=5)

    np.testing.assert_allclose(run.datasets()["relu@0"], first_call, rtol=1e-6)
    np.testing.assert_allclose(run.datasets()["relu@1"], second_call, rtol=1e-6)


def test_a_model_run_twice_per_item_is_refused_not_averaged(tmp_path):
    class TestTimeAugmentation(TensorConnector):
        def run_inference(self, model, batch):
            x = torch.stack(batch)
            with torch.no_grad():
                return (model(x) + model(x.flip(-1))) / 2

    run = white_box(tmp_path, value_frame(4), TestTimeAugmentation((1, 4, 4)), tiny_cnn(), layers=["conv"])

    assert not run.ok
    assert any("'conv' produced 8 rows for 4 items" in line and "more than once per item" in line
               for line in run.logs)


def test_grid_pool_matches_adaptive_average_pooling(tmp_path):
    model, conn, metadata = tiny_cnn(), TensorConnector((1, 6, 7)), value_frame(5)  # 6 x 7 does not divide by 4
    reference = capture_activation(model, "conv", conn.inputs_for(metadata))
    expected = torch.nn.functional.adaptive_avg_pool2d(torch.from_numpy(reference), 4).flatten(1).numpy()

    run = white_box(tmp_path, metadata, conn, model, layers={"conv": "Grid Pool"})

    assert run.datasets()["conv"].shape == (5, 3 * 16)
    np.testing.assert_allclose(run.datasets()["conv"], expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("shape", [(5,), (3, 7), (3, 6, 7), (2, 5, 4, 3)])
@pytest.mark.parametrize("strategy", ["Mean Pool", "Max Pool", "Grid Pool", "Seq Pool", "CLS Token", "Flatten"])
def test_stored_shape_says_what_the_engine_stores(shape, strategy):
    from prism.core.config import stored_shape
    from prism.core.execution import _prepare
    data = np.random.default_rng(0).normal(size=(2, *shape)).astype(np.float32)

    stored = _prepare("layer", data, 2, {"layer": strategy})

    assert stored_shape(shape, strategy) == stored.shape[1:]


def test_the_models_weights_are_fingerprinted_before_the_run_moves_it(tmp_path):
    model = tiny_cnn()
    expected = PyTorchInspector(model).model_fingerprint()

    run = white_box(tmp_path, value_frame(4), TensorConnector((1, 4, 4)), model, layers=["conv"])

    assert f"Model fingerprint: {expected}" in run.logs
    assert AuditCache.open(str(run.out_dir)).provenance["inputs"]["model"] == expected


def test_a_model_that_cannot_be_fingerprinted_is_still_extracted(tmp_path):
    class Unhashable(SklearnInspector):
        def model_fingerprint(self):
            raise TypeError("cannot pickle a database connection")

    run = run_engine(tmp_path / "out", value_frame(4), ValueConnector(), Unhashable(None))

    assert run.ok
    assert ("[WARNING] Could not fingerprint the model's weights (TypeError: cannot pickle a database "
            "connection).") in run.logs
    assert AuditCache.open(str(run.out_dir)).provenance["inputs"]["model"] is None


def test_each_layers_shape_is_logged_and_recorded_after_the_first_batch(tmp_path):
    model = tiny_cnn()

    run = white_box(tmp_path, value_frame(4), TensorConnector((1, 4, 4)), model,
                    layers={"conv": "Grid Pool", "fc": "Mean Pool"})

    assert "Layer 'conv': each row's output is 3×4×4; Grid Pool stores 48 values." in run.logs
    assert "Layer 'fc': 2 values per row, stored as they are." in run.logs
    cache = AuditCache.open(str(run.out_dir))
    assert cache.layer_summary("conv") == "3×4×4 → Grid Pool → 48 values"
    assert cache.layer_summary("fc") == "2 values"


def test_a_reused_modules_calls_are_summarised_together(tmp_path):
    model = ReusedActivation(conv=True).eval()

    run = white_box(tmp_path, value_frame(4), TensorConnector((2, 4, 4)), model, layers={"relu": "Max Pool"})

    assert AuditCache.open(str(run.out_dir)).layer_summary("relu@all") == "2 calls, each 2×4×4 → Max Pool → 2 values"


# ---------------------------------------------------------------------------
# Previewing layer shapes on one item (the Wiretap's Show shapes)
# ---------------------------------------------------------------------------

def test_preview_reports_each_layers_output_shape_from_one_row(tmp_path):
    from prism.core.execution import preview_shapes
    from prism.core.subject import Subject
    model, connector = tiny_cnn(), TensorConnector((1, 4, 4))

    shapes = preview_shapes(Subject(connector, model, PyTorchInspector(model)), value_frame(5), "subject_id",
                            ["conv", "fc"])

    assert shapes == {"conv": (3, 4, 4), "fc": (2,)}


def test_preview_names_each_call_of_a_reused_module(tmp_path):
    from prism.core.execution import preview_shapes
    from prism.core.subject import Subject
    model = ReusedActivation(conv=True).eval()

    shapes = preview_shapes(Subject(TensorConnector((2, 4, 4)), model, PyTorchInspector(model)), value_frame(3),
                            "subject_id", ["relu"])

    assert shapes == {"relu@0": (2, 4, 4), "relu@1": (2, 4, 4)}


def test_preview_leaves_the_model_as_it_was(tmp_path):
    from prism.core.execution import preview_shapes
    from prism.core.subject import Subject

    class BatchNormNet(nn.Module):  # batch norm in training mode refuses a batch of one
        def __init__(self):
            super().__init__()
            self.lin, self.norm = nn.Linear(4, 3), nn.BatchNorm1d(3)

        def forward(self, x):
            return self.norm(self.lin(x))

    model = BatchNormNet().train()
    inspector = PyTorchInspector(model)

    shapes = preview_shapes(Subject(TensorConnector((4,)), model, inspector), value_frame(2), "subject_id",
                            ["lin", "norm"])

    assert shapes == {"lin": (3,), "norm": (3,)}
    assert all(module.training for module in model.modules())  # back in training mode
    assert not model._forward_hooks and not model._forward_pre_hooks and not model.lin._forward_hooks


def test_preview_works_through_the_generic_inspector_path(tmp_path):
    from prism.core.execution import preview_shapes
    from prism.core.subject import Subject
    inspector = ScriptedInspector()

    shapes = preview_shapes(Subject(OneBadLayerConnector(inspector, bad_value=-1), None, inspector),
                            value_frame(2), "subject_id", ["good", "bad"])

    assert shapes == {"good": (1,), "bad": (1,)}


def test_grid_pool_on_3d_volumes():
    from prism.core.execution import _grid_pool
    volume = np.random.default_rng(0).normal(size=(2, 3, 9, 8, 10)).astype(np.float32)

    pooled = _grid_pool(volume)

    expected = torch.nn.functional.adaptive_avg_pool3d(torch.from_numpy(volume), 4).flatten(1).numpy()
    assert pooled.shape == (2, 3 * 64)
    np.testing.assert_allclose(pooled, expected, rtol=1e-5, atol=1e-6)


class ScriptedInspector:
    """Duck-typed inspector whose features are set by the connector, to inject a malformed batch."""

    extracts_hidden_layers = True

    def __init__(self):
        self.pending = {}

    def attach_hooks(self, layer_names):
        pass

    def retrieve_features(self, expected_rows=None):
        return dict(self.pending)

    def clear_buffer(self):
        self.pending = {}

    def teardown(self):
        pass


class OneBadLayerConnector(ValueConnector):
    def __init__(self, inspector, bad_value):
        super().__init__()
        self.inspector = inspector
        self.bad_value = bad_value

    def run_inference(self, model, batch):
        n_bad = len(batch) + 1 if self.bad_value in batch else len(batch)
        self.inspector.pending = {
            "good": np.asarray(batch, dtype=float)[:, None],
            "bad": np.zeros((n_bad, 1)),
        }
        return np.asarray(batch, dtype=float)


def test_a_layer_shape_mismatch_does_not_misalign_the_cache(tmp_path):
    inspector = ScriptedInspector()

    run = run_engine(tmp_path / "out", value_frame(9), OneBadLayerConnector(inspector, bad_value=4),
                     inspector, layers=["good", "bad"], batch_size=3, max_failed_percent=100)

    assert_aligned(run)
    assert list(run.failed_rows["value"]) == [3, 4, 5]
    assert set(run.failed_rows["prism_failed_at"]) == {"saving"}


def test_model_output_row_count_is_validated(tmp_path):
    class DropsAnOutput(ValueConnector):
        def run_inference(self, model, batch):
            out = super().run_inference(model, batch)
            return out[:-1] if 2 in batch else out

    run = black_box(tmp_path, value_frame(8), DropsAnOutput(), batch_size=4, max_failed_percent=100)

    assert_aligned(run)


def test_connector_receives_the_configured_id_column(tmp_path):
    metadata, conn = value_frame(5), ValueConnector()

    black_box(tmp_path, metadata, conn, id_col="subject_id")

    assert conn.seen_row_ids == list(metadata["subject_id"])


def test_numeric_looking_ids_reach_the_connector_and_cache_as_text(tmp_path):
    metadata = value_frame(3).assign(subject_id=["0042", "0043", "0100"])
    conn = ValueConnector()

    run = black_box(tmp_path, metadata, conn)

    assert conn.seen_row_ids == ["0042", "0043", "0100"]
    assert list(AuditCache.open(str(run.out_dir)).ids) == ["0042", "0043", "0100"]


def test_cache_records_the_run_and_the_connectors_class_order(tmp_path):
    class Declared(ValueConnector):
        class_names = ["healthy", "disease"]
        output_kind = "probabilities"

    run = black_box(tmp_path, value_frame(4), Declared(), group_col="label")
    cache = AuditCache.open(str(run.out_dir))

    assert (cache.config.id_col, cache.config.target_col, cache.config.group_col) == ("subject_id", "label", "label")
    assert (cache.class_names, cache.output_kind) == (["healthy", "disease"], "probabilities")


def test_a_missing_id_column_stops_the_run(tmp_path):
    run = black_box(tmp_path, value_frame(4).drop(columns="subject_id"), ValueConnector())

    assert not run.ok
    assert any("ID column 'subject_id' is not in the metadata" in line for line in run.logs)


def test_repeated_ids_stop_the_run(tmp_path):
    metadata = value_frame(4).assign(subject_id=["S1", "S2", "S1", "S2"])

    run = black_box(tmp_path, metadata, ValueConnector())

    assert not run.ok
    assert any("must identify rows uniquely; repeated values: S1, S2" in line for line in run.logs)


def test_a_layer_that_stops_appearing_skips_that_batch(tmp_path):
    class DropsALayer(OneBadLayerConnector):
        def run_inference(self, model, batch):
            out = super().run_inference(model, batch)
            if 4 in batch:
                del self.inspector.pending["good"]
            return out

    inspector = ScriptedInspector()
    run = run_engine(tmp_path / "out", value_frame(9), DropsALayer(inspector, bad_value=-1),
                     inspector, layers=["good", "bad"], batch_size=3, max_failed_percent=100)

    assert run.ok
    assert list(run.metadata["value"]) == [0, 1, 2, 6, 7, 8]
    assert any("Batch 1 skipped" in line and "missing ['good']" in line for line in run.logs)
    assert_aligned(run)


# ---------------------------------------------------------------------------
# Correctness bundle: eval mode, no autograd, no silent overwrite
# ---------------------------------------------------------------------------

class DropoutNet(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.lin = nn.Linear(4, 4)
        self.drop = nn.Dropout(0.5)
        self.head = nn.Linear(4, 2)

    def forward(self, x):
        return self.head(self.drop(self.lin(x)))


def test_a_model_left_in_training_mode_is_probed_in_eval_mode(tmp_path):
    model, conn, metadata = DropoutNet().train(), TensorConnector((4,)), value_frame(8)
    model.eval()
    reference = capture_activation(model, "drop", conn.inputs_for(metadata))
    model.train()

    first = run_engine(tmp_path / "a", metadata, conn, PyTorchInspector(model), model=model, layers=["drop"])
    second = run_engine(tmp_path / "b", metadata, conn, PyTorchInspector(model), model=model, layers=["drop"])

    np.testing.assert_allclose(first.datasets()["drop"], reference, rtol=1e-6)  # dropout was off
    np.testing.assert_array_equal(first.datasets()["drop"], second.datasets()["drop"])
    assert any("[WARNING] The model was in training mode" in line for line in first.logs)
    assert model.training and model.drop.training  # the model is handed back as it was


def test_batch_norm_uses_its_stored_statistics_whatever_the_batch(tmp_path):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4)).train()
    conn, metadata = TensorConnector((3,)), value_frame(8)

    one = run_engine(tmp_path / "bs1", metadata, conn, PyTorchInspector(model), model=model, layers=["1"], batch_size=1)
    four = run_engine(tmp_path / "bs4", metadata, conn, PyTorchInspector(model), model=model, layers=["1"], batch_size=4)

    assert len(one.metadata) == 8 and len(four.metadata) == 8  # training-mode batch norm rejects batches of one
    np.testing.assert_allclose(one.datasets()["1"], four.datasets()["1"], rtol=1e-6)


def test_inference_runs_without_autograd(tmp_path):
    class ForgetsNoGrad(TensorConnector):
        def run_inference(self, model, batch):
            self.grad_enabled = torch.is_grad_enabled()
            return model(torch.stack(batch))

    conn = ForgetsNoGrad((1, 4, 4))

    run = white_box(tmp_path, value_frame(4), conn, tiny_cnn(), layers=["fc"])

    assert run.ok and conn.grad_enabled is False


def test_a_model_already_in_eval_mode_gets_no_warning(tmp_path):
    run = white_box(tmp_path, value_frame(4), TensorConnector((1, 4, 4)), tiny_cnn(), layers=["fc"])

    assert not any("training mode" in line for line in run.logs)


def test_existing_results_are_not_replaced_silently(tmp_path):
    first = black_box(tmp_path, value_frame(4), ValueConnector())

    second = black_box(tmp_path, value_frame(6), ValueConnector())

    assert first.ok and not second.ok
    assert any("Results already exist" in line for line in second.logs)
    assert len(second.metadata) == 4  # the first run's results are untouched


def test_overwrite_replaces_existing_results(tmp_path):
    black_box(tmp_path, value_frame(4), ValueConnector())

    run = black_box(tmp_path, value_frame(6), ValueConnector(), overwrite=True)

    assert run.ok and len(run.metadata) == 6


def test_a_replacing_run_that_fails_keeps_the_earlier_results(tmp_path):
    black_box(tmp_path, value_frame(4), ValueConnector())

    run = black_box(tmp_path, value_frame(6), ValueConnector(fail_load=set(range(6))), overwrite=True)

    assert not run.ok
    assert run.logs[-1].endswith("The results already in the output directory are unchanged.")
    assert len(AuditCache.open(str(run.out_dir)).metadata) == 4
    assert sorted(p.name for p in run.out_dir.iterdir()) == [H5_NAME, "prism_metadata.csv"]


def test_a_replacing_run_that_crashes_keeps_the_earlier_results(tmp_path):
    black_box(tmp_path, value_frame(4), ValueConnector())

    def explode(value):
        raise RuntimeError("progress callback failed")

    run = black_box(tmp_path, value_frame(8), ValueConnector(), overwrite=True, progress_callback=explode)

    assert not run.ok
    assert ("[CRITICAL ERROR] progress callback failed The results already in the output directory are "
            "unchanged.") in run.logs
    assert len(AuditCache.open(str(run.out_dir)).metadata) == 4


def test_a_cancelled_replacing_run_keeps_the_earlier_results(tmp_path):
    from prism.core.cancel import Cancelled
    black_box(tmp_path, value_frame(4), ValueConnector())

    def cancel(value):
        raise Cancelled()

    with pytest.raises(Cancelled):
        black_box(tmp_path, value_frame(8), ValueConnector(), overwrite=True, progress_callback=cancel)

    assert len(AuditCache.open(str(tmp_path / "out")).metadata) == 4



# ---------------------------------------------------------------------------
# Scale safeguards
# ---------------------------------------------------------------------------

def test_the_projected_size_is_logged(tmp_path):
    run = black_box(tmp_path, value_frame(8), ValueConnector(), batch_size=4)

    assert any(line.startswith("Projected size of the results:") and "for 8 rows" in line for line in run.logs)


def test_a_run_that_would_fill_the_disk_stops_after_the_first_batch(tmp_path, monkeypatch):
    import prism.core.execution as execution_module
    from types import SimpleNamespace
    monkeypatch.setattr(execution_module.shutil, "disk_usage", lambda path: SimpleNamespace(free=100))

    run = white_box(tmp_path, value_frame(8), TensorConnector((1, 4, 4)), tiny_cnn(), layers={"conv": "Flatten"})

    assert not run.ok
    assert any("Not enough disk space" in line and "Grid Pool" in line for line in run.logs)
    assert not (run.out_dir / H5_NAME).exists()  # the half-written file is removed


def test_a_warning_when_the_results_take_over_half_the_free_space(tmp_path, monkeypatch):
    import prism.core.execution as execution_module
    from types import SimpleNamespace
    monkeypatch.setattr(execution_module.shutil, "disk_usage", lambda path: SimpleNamespace(free=200))

    run = black_box(tmp_path, value_frame(8), ValueConnector(), batch_size=4)  # 8 rows x 8 bytes = 64 B... of 200

    assert run.ok
    assert not any("more than half" in line for line in run.logs)
    monkeypatch.setattr(execution_module.shutil, "disk_usage", lambda path: SimpleNamespace(free=100))
    again = black_box(tmp_path, value_frame(8), ValueConnector(), batch_size=4, overwrite=True)
    assert again.ok and any("more than half of the free disk space" in line for line in again.logs)



# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------

def test_a_cancelled_run_stops_cleans_up_and_says_so(tmp_path):
    from prism.core.cancel import Cancelled
    from prism.core.config import AuditConfig
    from prism.core.execution import run_extraction
    from prism.core.subject import Subject
    connector, logs = ValueConnector(), []
    out = tmp_path / "out"
    out.mkdir()
    config = AuditConfig(project_root="", script_path="", framework="in-process", csv_path="",
                         id_col="subject_id", target_col="label", layers={}, batch_size=2, output_dir=str(out))

    def check():
        if len(connector.calls) == 5:  # stop while loading the third batch
            raise Cancelled()

    with pytest.raises(Cancelled):
        run_extraction(config, Subject(connector, None, SklearnInspector(None)), value_frame(8),
                       lambda value: None, logs.append, check=check)

    assert len(connector.calls) == 5
    assert logs[-1] == "[CANCELLED] Extraction stopped; nothing was saved."
    assert not (out / H5_NAME).exists()  # the half-written file is removed, so the next run need not ask


def test_cancelling_from_the_progress_callback_works_too(tmp_path):
    from prism.core.cancel import Cancelled

    def cancel(value):
        raise Cancelled()

    with pytest.raises(Cancelled):
        black_box(tmp_path, value_frame(4), ValueConnector(), progress_callback=cancel)
