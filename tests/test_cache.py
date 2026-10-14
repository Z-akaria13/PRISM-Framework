"""core.cache: the results files every analysis tab reads."""
import h5py
import numpy as np
import pandas as pd
import pytest

from _support import write_legacy_cache
from prism.core.cache import (CSV_NAME, FAILED_NAME, H5_NAME, IDS, LEGACY_NAMES, MODEL_OUTPUT, PARTIAL, AuditCache,
                              BatchMismatch, CacheError, CacheWriter)
from prism.core.config import AuditConfig


def audit_config(out_dir):
    return AuditConfig(project_root="/proj", script_path="/proj/c.py", framework="PyTorch",
                       csv_path="/proj/meta.csv", id_col="subject_id", target_col="label",
                       group_col="patient", output_dir=str(out_dir), layers={"conv": "Max Pool"})


def cohort(n=4, ids=None):
    return pd.DataFrame({
        "subject_id": ids if ids is not None else [f"S{i:03d}" for i in range(n)],
        "label": [i % 2 for i in range(n)],
        "patient": [i // 2 for i in range(n)],
    })


def batch(rows):
    rows = np.asarray(rows)
    return {MODEL_OUTPUT: rows / 10, "conv": np.repeat(rows[:, None], 3, axis=1).astype(float)}


def write(out_dir, metadata=None, class_names=None, output_kind=None):
    metadata = cohort() if metadata is None else metadata
    writer = CacheWriter(str(out_dir), audit_config(out_dir), class_names=class_names, output_kind=output_kind)
    half = len(metadata) // 2
    writer.append(metadata["subject_id"][:half], batch(range(half)))
    writer.append(metadata["subject_id"][half:], batch(range(half, len(metadata))))
    writer.finish(metadata)
    return metadata


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------

def test_round_trip(tmp_path):
    metadata = write(tmp_path, class_names=["neg", "pos"], output_kind="probabilities")

    cache = AuditCache.open(str(tmp_path))

    assert not cache.legacy and cache.warnings == []
    assert list(cache.ids) == list(metadata["subject_id"])
    assert cache.metadata.equals(metadata)
    assert cache.config == audit_config(tmp_path)
    assert cache.layer_names == ["conv"]
    assert (cache.class_names, cache.output_kind) == (["neg", "pos"], "probabilities")
    np.testing.assert_allclose(cache.model_output(), [0.0, 0.1, 0.2, 0.3])
    np.testing.assert_allclose(cache.layer("conv")[:, 0], [0, 1, 2, 3])
    assert cache.layer_nbytes("conv") == 4 * 3 * 8


def test_layers_record_their_pooling_and_the_file_is_marked_complete(tmp_path):
    write(tmp_path)

    with h5py.File(tmp_path / H5_NAME, "r") as f:
        assert f["conv"].attrs["pooling"] == "Max Pool"
        assert "pooling" not in f[MODEL_OUTPUT].attrs
        assert f.attrs["complete"]


def test_unknown_layer_is_a_key_error(tmp_path):
    write(tmp_path)

    with pytest.raises(KeyError, match="not found"):
        AuditCache.open(str(tmp_path)).layer("fc")


@pytest.mark.parametrize("ids", [["0042", "0043", "0100", "1000"], [42, 43, 100, 1000], [1.5, 2.0, 3.0, 4.25]])
def test_ids_that_pandas_would_reinterpret_still_match(tmp_path, ids):
    metadata = cohort(ids=ids)
    writer = CacheWriter(str(tmp_path), audit_config(tmp_path))
    writer.append(metadata["subject_id"].astype(str), batch(range(4)))
    writer.finish(metadata)

    cache = AuditCache.open(str(tmp_path))

    assert list(cache.ids) == [str(i) for i in ids]


# ---------------------------------------------------------------------------
# All-or-nothing batches
# ---------------------------------------------------------------------------

@pytest.fixture
def writer(tmp_path):
    w = CacheWriter(str(tmp_path), audit_config(tmp_path))
    w.append(["S000", "S001"], batch([0, 1]))
    yield w
    w.abort()


def lengths(tmp_path, writer):
    writer.finish(cohort(2))  # the metadata of the first batch's two rows
    with h5py.File(tmp_path / H5_NAME, "r") as f:
        return {name: f[name].shape[0] for name in f.keys()}


@pytest.mark.parametrize("bad_batch, message", [
    ({MODEL_OUTPUT: np.zeros(2), "conv": np.zeros((3, 3))}, "expected 2 rows"),
    ({MODEL_OUTPUT: np.zeros(2)}, "missing \\['conv'\\]"),
    ({**batch([2, 3]), "fc": np.zeros((2, 2))}, "new \\['fc'\\]"),
    ({MODEL_OUTPUT: np.zeros(2), "conv": np.zeros((2, 4))}, "feature shape changed"),
    ({MODEL_OUTPUT: np.float64(1.0), "conv": np.zeros((2, 3))}, "a scalar"),
])
def test_an_inconsistent_batch_is_rejected_without_writing_anything(tmp_path, writer, bad_batch, message):
    with pytest.raises(BatchMismatch, match=message):
        writer.append(["S002", "S003"], bad_batch)

    assert lengths(tmp_path, writer) == {IDS: 2, MODEL_OUTPUT: 2, "conv": 2}


def test_abort_releases_the_file(tmp_path, writer):
    writer.abort()

    h5py.File(tmp_path / (H5_NAME + PARTIAL), "w").close()


# ---------------------------------------------------------------------------
# Replacing results only when the new ones are complete (debt P1 #24)
# ---------------------------------------------------------------------------

def test_existing_results_stay_in_place_while_a_new_run_writes(tmp_path):
    old = write(tmp_path)
    writer = CacheWriter(str(tmp_path), audit_config(tmp_path))
    writer.append(["N000"], batch([9]))

    assert list(AuditCache.open(str(tmp_path)).ids) == list(old["subject_id"])
    writer.abort()
    assert list(AuditCache.open(str(tmp_path)).ids) == list(old["subject_id"])
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([CSV_NAME, H5_NAME])


def test_finish_replaces_the_results_and_leaves_no_partial_files(tmp_path):
    write(tmp_path)
    new = cohort(2, ids=["N000", "N001"])
    writer = CacheWriter(str(tmp_path), audit_config(tmp_path))
    writer.append(new["subject_id"], batch([5, 6]))

    writer.finish(new)

    assert list(AuditCache.open(str(tmp_path)).ids) == ["N000", "N001"]
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([CSV_NAME, H5_NAME])


def test_an_earlier_runs_failed_rows_and_old_names_go_only_when_the_new_run_finishes(tmp_path):
    leftovers = [FAILED_NAME, *LEGACY_NAMES]
    for name in leftovers:
        (tmp_path / name).write_text("from an earlier run")
    writer = CacheWriter(str(tmp_path), audit_config(tmp_path))
    writer.append(["S000", "S001"], batch([0, 1]))
    assert all((tmp_path / name).exists() for name in leftovers)

    writer.finish(cohort(2))

    assert not any((tmp_path / name).exists() for name in leftovers)


# ---------------------------------------------------------------------------
# Failed rows (debt P0 #22)
# ---------------------------------------------------------------------------

def write_with_failures(out_dir):
    """Rows S001 and S003 (both label 1) failed to load; S000 and S002 were extracted."""
    metadata = cohort(4)
    failed = metadata.iloc[[1, 3]].reset_index(drop=True).assign(prism_failed_at="load_item",
                                                                prism_error="OSError: gone")
    writer = CacheWriter(str(out_dir), audit_config(out_dir))
    writer.append(["S000", "S002"], batch([0, 2]))
    writer.finish(metadata.iloc[[0, 2]].reset_index(drop=True), failed, rows_attempted=4)


def test_failed_rows_are_listed_counted_and_described(tmp_path):
    write_with_failures(tmp_path)

    cache = AuditCache.open(str(tmp_path))

    assert (cache.rows_attempted, cache.rows_failed, len(cache.metadata)) == (4, 2, 2)
    assert list(cache.failures["subject_id"]) == ["S001", "S003"]
    assert cache.failure_note == (
        "2 of 4 rows (50%) failed during the extraction and are not in these results: 2 in load_item "
        "(first error: OSError: gone). Most affected: label = 1 (all 2 rows: missing from every analysis). "
        "prism_failed_rows.csv lists each one with its error.")
    assert cache.warnings == [cache.failure_note]
    with h5py.File(tmp_path / H5_NAME, "r") as f:
        assert (f.attrs["schema_version"], f.attrs["rows_attempted"], f.attrs["rows_failed"]) == (3, 4, 2)


def test_layers_written_before_shapes_were_recorded_have_no_summary(tmp_path):
    write(tmp_path)

    assert AuditCache.open(str(tmp_path)).layer_summary("conv") is None


def test_results_without_failures_say_nothing_about_them(tmp_path):
    write(tmp_path)

    cache = AuditCache.open(str(tmp_path))

    assert (cache.rows_attempted, cache.rows_failed, cache.failures, cache.failure_note) == (4, 0, None, None)


@pytest.mark.parametrize("damage", ["delete", "truncate"])
def test_a_missing_or_mismatched_list_of_failed_rows_is_still_counted(tmp_path, damage):
    write_with_failures(tmp_path)
    path = tmp_path / FAILED_NAME
    if damage == "delete":
        path.unlink()
    else:
        pd.read_csv(path).iloc[:1].to_csv(path, index=False)

    cache = AuditCache.open(str(tmp_path))

    assert cache.rows_failed == 2 and cache.failures is None
    assert cache.failure_note == ("2 of 4 rows (50%) failed during the extraction and are not in these results. "
                                  "prism_failed_rows.csv, which should list them, is missing or from another run.")


# ---------------------------------------------------------------------------
# Opening defensively
# ---------------------------------------------------------------------------

def test_missing_files_are_reported(tmp_path):
    with pytest.raises(CacheError, match="Cache files not found"):
        AuditCache.open(str(tmp_path))


def test_a_cache_without_predictions_is_rejected(tmp_path):
    with h5py.File(tmp_path / H5_NAME, "w") as f:
        f.create_dataset("conv", data=np.zeros((2, 3)))
    cohort(2).to_csv(tmp_path / CSV_NAME, index=False)

    with pytest.raises(CacheError, match="Final predictions missing"):
        AuditCache.open(str(tmp_path))


@pytest.mark.parametrize("edit", [
    lambda df: df.iloc[::-1],       # re-sorted
    lambda df: df.iloc[1:],         # a row deleted
    lambda df: df.drop(columns="subject_id"),
])
def test_an_edited_metadata_csv_is_rejected(tmp_path, edit):
    metadata = write(tmp_path)
    edit(metadata).to_csv(tmp_path / CSV_NAME, index=False)

    with pytest.raises(CacheError, match="no longer matches the row IDs"):
        AuditCache.open(str(tmp_path))


def test_legacy_caches_open_with_a_warning(tmp_path):
    write_legacy_cache(tmp_path, cohort(), np.zeros(4), {"conv": np.zeros((4, 3))})

    cache = AuditCache.open(str(tmp_path))

    assert cache.legacy and cache.ids is None and cache.config is None and cache.class_names is None
    assert cache.layer_names == ["conv"]
    assert "matched to the metadata CSV by position" in cache.warnings[0]


def test_misaligned_layers_in_legacy_caches_are_left_out(tmp_path):
    write_legacy_cache(tmp_path, cohort(), np.zeros(4), {"conv": np.zeros((4, 3)), "short": np.zeros((3, 2))})

    cache = AuditCache.open(str(tmp_path))

    assert cache.layer_names == ["conv"]
    assert any("Layer 'short' has 3 rows for 4 metadata rows" in w for w in cache.warnings)
    with pytest.raises(KeyError):
        cache.layer("short")


def test_misaligned_predictions_are_rejected(tmp_path):
    write_legacy_cache(tmp_path, cohort(), np.zeros(3))

    with pytest.raises(CacheError, match="3 predictions for 4 metadata rows"):
        AuditCache.open(str(tmp_path))


def test_open_layer_reads_rows_on_demand(tmp_path):
    write(tmp_path)
    cache = AuditCache.open(str(tmp_path))

    with cache.open_layer("conv") as layer:
        assert layer.shape == (4, 3)
        np.testing.assert_allclose(layer[1:3, 0], [1, 2])
    with pytest.raises(KeyError, match="not found"):
        with cache.open_layer("fc"):
            pass



def test_calls_of_a_reused_module_also_read_side_by_side(tmp_path):
    config = AuditConfig(project_root="", script_path="", framework="t", csv_path="", id_col="subject_id",
                         target_col="label", output_dir=str(tmp_path), layers={"relu": "Mean Pool"})
    metadata = cohort()
    writer = CacheWriter(str(tmp_path), config)
    writer.append(metadata["subject_id"], {MODEL_OUTPUT: np.zeros(4), "relu@0": np.ones((4, 2)),
                                           "relu@1": np.full((4, 3), 2.0)})
    writer.finish(metadata)

    cache = AuditCache.open(str(tmp_path))

    assert cache.layer_names == ["relu@0", "relu@1", "relu@all"]
    assert cache.calls_of("relu@all") == ["relu@0", "relu@1"]
    np.testing.assert_array_equal(cache.layer("relu@all"), np.hstack([np.ones((4, 2)), np.full((4, 3), 2.0)]))
    assert cache.layer_nbytes("relu@all") == (8 + 12) * 8
    with cache.open_layer("relu@all") as both:
        assert both.shape == (4, 5)
        np.testing.assert_array_equal(both[1:3], [[1, 1, 2, 2, 2]] * 2)


def test_module_of_reads_call_suffixes_only():
    from prism.core.cache import module_of

    assert module_of("layer1.0.relu@1") == "layer1.0.relu"
    assert module_of("layer1.0.relu@all") == "layer1.0.relu"
    assert module_of("conv") == "conv"
    assert module_of("odd@name") == "odd@name"



def test_abort_removes_a_half_written_file(tmp_path, writer):
    writer.abort()

    assert not (tmp_path / H5_NAME).exists()


def test_abort_after_finish_keeps_the_results(tmp_path, writer):
    writer.finish(cohort(2))
    writer.abort()

    assert len(AuditCache.open(str(tmp_path)).metadata) == 2


@pytest.mark.parametrize("shape, dtype, expected", [
    ((10, 128), np.float32, (2048, 128)),       # 512-byte rows: 2048 of them per ~1 MB chunk
    ((10,), np.float32, (262144,)),             # model output
    ((3, 2**20), np.float32, (1, 262144)),      # a 4 MB row is split into ~1 MB pieces
    ((10, 4, 2**18), np.float64, (1, 4, 32768)),
])
def test_chunks_are_row_blocks_of_about_a_megabyte(shape, dtype, expected):
    from prism.core.cache import _row_chunks

    assert _row_chunks(np.zeros(shape, dtype=dtype)) == expected


def test_written_layers_use_row_block_chunks(tmp_path):
    write(tmp_path)

    with h5py.File(tmp_path / H5_NAME, "r") as f:
        assert f["conv"].chunks == (2**20 // 24, 3)
