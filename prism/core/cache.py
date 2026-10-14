"""
RESULTS CACHE
What an extraction leaves in its output directory, and the only thing the
analysis tabs read:

    prism_embeddings.h5   one dataset per selected layer, "__model_output__",
                          "__ids__" (every row's ID-column value), and attributes
                          recording the run's AuditConfig and the model's class order
    prism_metadata.csv    the metadata rows that made it into the cache, same order
    prism_failed_rows.csv the metadata rows that failed, with where and why (only if any did)

Results written before PRISM was renamed (whydunnit_embeddings.h5 / whydunnit_metadata.csv)
open the same way.

Batches are written all-or-nothing, so every dataset always has one row per CSV row.
AuditCache.open() re-checks that, and the IDs, before anything gets analysed.

A run writes to "<name>.partial" files and moves them into place only when it finishes, so
results already in the directory survive a run that fails or is cancelled.
"""
import json
import os
from contextlib import contextmanager

import h5py
import numpy as np
import pandas as pd

from .config import AuditConfig, shape_text
from .failures import ERROR, FAILED_AT, describe, percent
from .metadata import read_metadata

H5_NAME = "prism_embeddings.h5"
CSV_NAME = "prism_metadata.csv"
FAILED_NAME = "prism_failed_rows.csv"
LEGACY_NAMES = ("whydunnit_embeddings.h5", "whydunnit_metadata.csv")  # written before the rename; still read
PARTIAL = ".partial"  # suffix of a run's files until it finishes
MODEL_OUTPUT = "__model_output__"
IDS = "__ids__"
SCHEMA_VERSION = 3  # 2: rows_attempted / rows_failed and prism_failed_rows.csv; 3: provenance
CHUNK_TARGET_BYTES = 2**20  # HDF5 chunks of about 1 MB: whole rows, so reading rows touches few chunks


ALL_CALLS = "all"  # "<module>@all": every stored call of a reused module, side by side


def module_of(name):
    """The module a dataset belongs to: "layer1.0.relu@1" -> "layer1.0.relu"; plain names unchanged."""
    module, sep, call = name.rpartition("@")
    return module if sep and (call.isdigit() or call == ALL_CALLS) else name


def results_exist(output_dir):
    """Whether `output_dir` already holds (possibly partial) extraction results, under either name."""
    return any(os.path.exists(os.path.join(output_dir, name)) for name in (H5_NAME, CSV_NAME, *LEGACY_NAMES))


def result_files(output_dir):
    """(HDF5 path, CSV path) of the complete results in `output_dir`, current names first; None if absent."""
    for h5_name, csv_name in ((H5_NAME, CSV_NAME), LEGACY_NAMES):
        h5_path, csv_path = os.path.join(output_dir, h5_name), os.path.join(output_dir, csv_name)
        if os.path.exists(h5_path) and os.path.exists(csv_path):
            return h5_path, csv_path
    return None


class CacheError(Exception):
    """The directory does not hold a cache that can be analysed safely."""


class BatchMismatch(ValueError):
    """A batch disagrees with itself or with earlier batches. Nothing from it was written."""


class CacheWriter:
    """Writes one run's results to .partial files; finish() moves them into place."""

    def __init__(self, output_dir, config, class_names=None, output_kind=None):
        self.output_dir = output_dir
        self.rows = 0
        self._finished = False
        self._pooling = dict(config.layers)
        self._datasets = {}
        self._file = h5py.File(self._partial(H5_NAME), "w")
        attrs = self._file.attrs
        attrs["schema_version"] = SCHEMA_VERSION
        attrs["audit_config"] = config.to_json()
        if class_names is not None:
            attrs["class_names"] = json.dumps([str(c) for c in class_names])
        if output_kind is not None:
            attrs["output_kind"] = str(output_kind)
        self._ids = self._file.create_dataset(IDS, shape=(0,), maxshape=(None,),
                                              dtype=h5py.string_dtype(), chunks=True)

    def append(self, ids, arrays):
        """Add one batch: `ids` (one per row) and {name: array with one row per id}."""
        ids = [str(i) for i in ids]
        n = len(ids)
        arrays = {name: np.asarray(a) for name, a in arrays.items()}

        wrong = {name: (a.shape[0] if a.ndim else "a scalar") for name, a in arrays.items()
                 if a.ndim == 0 or a.shape[0] != n}
        if wrong:
            raise BatchMismatch(f"expected {n} rows, got {wrong}")
        if self._datasets:
            missing = sorted(set(self._datasets) - set(arrays))
            extra = sorted(set(arrays) - set(self._datasets))
            if missing or extra:
                raise BatchMismatch(f"outputs differ from earlier batches (missing {missing}, new {extra})")
            reshaped = {name: (self._datasets[name].shape[1:], a.shape[1:]) for name, a in arrays.items()
                        if self._datasets[name].shape[1:] != a.shape[1:]}
            if reshaped:
                raise BatchMismatch(f"feature shape changed from earlier batches: {reshaped}")

        for name, a in arrays.items():
            if name not in self._datasets:
                dset = self._file.create_dataset(name, data=a, maxshape=(None, *a.shape[1:]), chunks=_row_chunks(a))
                module = module_of(name)
                if module in self._pooling:
                    dset.attrs["pooling"] = self._pooling[module]
                if module != name:
                    dset.attrs["module"], dset.attrs["call"] = module, int(name.rpartition("@")[2])
                self._datasets[name] = dset
            else:
                dset = self._datasets[name]
                dset.resize(dset.shape[0] + n, axis=0)
                dset[-n:] = a
        self._ids.resize(self.rows + n, axis=0)
        self._ids[self.rows:] = ids
        self.rows += n

    def note_output_shapes(self, shapes):
        """Record {dataset: shape of one row's output before pooling} on the datasets."""
        for name, shape in shapes.items():
            if name in self._datasets:
                self._datasets[name].attrs["output_shape"] = list(shape)

    def finish(self, metadata, failed=None, rows_attempted=None, provenance=None):
        """Write the metadata of the appended rows (in order) and of the rows that failed (with
        FAILED_AT and ERROR), and what produced them (core/provenance.py), then move this run's
        files into place, replacing earlier results."""
        failed = failed if failed is not None and len(failed) else None
        n_failed = 0 if failed is None else len(failed)
        attrs = self._file.attrs
        attrs["rows_attempted"] = int(rows_attempted if rows_attempted is not None else self.rows + n_failed)
        attrs["rows_failed"] = n_failed
        if provenance is not None:
            attrs["provenance"] = json.dumps(provenance)
        attrs["complete"] = True
        self._file.close()
        metadata.to_csv(self._partial(CSV_NAME), index=False)
        if failed is not None:
            failed.to_csv(self._partial(FAILED_NAME), index=False)

        # The HDF5 file first: if this is interrupted part-way, its IDs disagree with the old CSV,
        # which open() refuses, rather than pairing new activations with old rows.
        os.replace(self._partial(H5_NAME), self._path(H5_NAME))
        os.replace(self._partial(CSV_NAME), self._path(CSV_NAME))
        if failed is not None:
            os.replace(self._partial(FAILED_NAME), self._path(FAILED_NAME))
        stale = ([] if failed is not None else [FAILED_NAME]) + list(LEGACY_NAMES)
        for name in stale:  # an earlier run's list of failed rows, or results under the old names
            if os.path.exists(self._path(name)):
                os.remove(self._path(name))
        self._finished = True

    def abort(self):
        """Close and delete this run's .partial files. Results already in the directory are left as
        they were. Does nothing once finish() has written the results."""
        if self._finished:
            return
        if self._file.id.valid:
            self._file.close()
        for name in (H5_NAME, CSV_NAME, FAILED_NAME):
            if os.path.exists(self._partial(name)):
                os.remove(self._partial(name))

    def _path(self, name):
        return os.path.join(self.output_dir, name)

    def _partial(self, name):
        return self._path(name) + PARTIAL


def _row_chunks(array):
    """Chunk shape for a dataset that grows by rows: blocks of whole rows of about
    CHUNK_TARGET_BYTES, or for rows wider than that, one row split into ~1 MB pieces."""
    row_bytes = array.itemsize * int(np.prod(array.shape[1:]))
    if array.ndim == 1 or row_bytes <= CHUNK_TARGET_BYTES:
        return (max(1, CHUNK_TARGET_BYTES // max(row_bytes, 1)),) + array.shape[1:]
    inner = array.itemsize * int(np.prod(array.shape[1:-1]))
    return (1,) + array.shape[1:-1] + (max(1, min(array.shape[-1], CHUNK_TARGET_BYTES // inner)),)


class AuditCache:
    """A cache opened for analysis. Use AuditCache.open(output_dir)."""

    def __init__(self, output_dir, metadata, config, ids, layer_names, class_names, output_kind, warnings,
                 h5_path=None, failures=None, rows_attempted=None, failure_note=None, provenance=None):
        self.output_dir = output_dir
        self._h5_path = h5_path or os.path.join(output_dir, H5_NAME)
        self.metadata = metadata
        self.config = config            # the run's AuditConfig, or None for legacy caches
        self.ids = ids                  # every row's ID as a string, or None for legacy caches
        self.layer_names = layer_names  # layers that can be analysed (aligned with metadata)
        self.class_names = class_names  # model output order declared by the connector, or None
        self.output_kind = output_kind
        self.warnings = warnings        # things the user should know before trusting the results
        self.failures = failures        # the rows that failed (prism_failed_rows.csv), or None
        self.rows_attempted = rows_attempted if rows_attempted is not None else len(metadata)
        self.failure_note = failure_note  # the warning that describes them, or None
        self.provenance = provenance      # what produced the results (core/provenance.py), or None if not recorded

    @property
    def rows_failed(self):
        return self.rows_attempted - len(self.metadata)

    @property
    def legacy(self):
        return self.ids is None

    @property
    def h5_path(self):
        return self._h5_path

    @classmethod
    def exists(cls, output_dir):
        return bool(output_dir) and result_files(output_dir) is not None

    @classmethod
    def open(cls, output_dir):
        if not output_dir or not cls.exists(output_dir):
            raise CacheError(f"Cache files not found in:\n{output_dir}")
        h5_path, csv_path = result_files(output_dir)

        with h5py.File(h5_path, "r") as f:
            attrs = dict(f.attrs)
            lengths = {name: f[name].shape[0] for name in f.keys()}
            ids = f[IDS].asstr()[:] if IDS in f else None

        if MODEL_OUTPUT not in lengths:
            raise CacheError("Final predictions missing from cache.")

        warnings = []
        config = AuditConfig.from_json(attrs["audit_config"]) if "audit_config" in attrs else None
        if ids is None:
            metadata = pd.read_csv(csv_path)
            warnings.append("This cache predates row IDs, so its rows are matched to the metadata CSV by position.")
        else:
            metadata = read_metadata(csv_path, config.id_col)
            csv_ids = metadata[config.id_col].fillna("nan").tolist() if config.id_col in metadata else None
            if csv_ids != list(ids):
                raise CacheError(f"{os.path.basename(csv_path)} no longer matches the row IDs recorded in the cache "
                                 "(was it edited or replaced?). Re-run the extraction.")

        n = len(metadata)
        if lengths[MODEL_OUTPUT] != n:
            raise CacheError(f"The cache holds {lengths[MODEL_OUTPUT]} predictions for {n} metadata rows, "
                             "so they cannot be matched. Re-run the extraction.")

        layer_names = []
        for name, length in lengths.items():
            if name.startswith("__"):
                continue
            if length != n:
                warnings.append(f"Layer '{name}' has {length} rows for {n} metadata rows and has been left out. "
                                "Caches from older versions could misalign a layer; re-extract to use it.")
                continue
            layer_names.append(name)
        calls = _calls_by_module(layer_names)
        for module in [m for m, names in calls.items() if len(names) > 1]:
            layer_names.insert(layer_names.index(calls[module][-1]) + 1, f"{module}@{ALL_CALLS}")

        failures, rows_attempted, failure_note = _failed_rows(output_dir, attrs, metadata, config, warnings)
        class_names = json.loads(attrs["class_names"]) if "class_names" in attrs else None
        return cls(output_dir, metadata, config, ids, layer_names, class_names,
                   attrs.get("output_kind"), warnings, h5_path=h5_path, failures=failures,
                   rows_attempted=rows_attempted, failure_note=failure_note,
                   provenance=json.loads(attrs["provenance"]) if "provenance" in attrs else None)

    def calls_of(self, name):
        """The stored datasets behind `name`: itself, or every call of the module for "<module>@all"."""
        if name.endswith(f"@{ALL_CALLS}"):
            return _calls_by_module(self.layer_names)[module_of(name)]
        return [name]

    def model_output(self):
        with h5py.File(self.h5_path, "r") as f:
            return f[MODEL_OUTPUT][:]

    def layer(self, name):
        with self.open_layer(name) as data:
            return data[:]

    @contextmanager
    def open_layer(self, name):
        """The layer as an open on-disk h5py dataset. Slicing it reads only those rows, so
        streamed analyses (e.g. the ridge probe) never load the whole layer."""
        self._check_layer(name)
        with h5py.File(self.h5_path, "r") as f:
            datasets = [f[call] for call in self.calls_of(name)]
            yield datasets[0] if len(datasets) == 1 else _SideBySide(datasets)

    def layer_summary(self, name):
        """'64×28×28 → Mean Pool → 64' for a layer, from what its run recorded; None for results
        written before output shapes were recorded."""
        parts = []
        with h5py.File(self.h5_path, "r") as f:
            for call in self.calls_of(name):
                attrs = f[call].attrs
                if "output_shape" not in attrs:
                    return None
                stored = shape_text(f[call].shape[1:])
                parts.append(f"{shape_text(attrs['output_shape'])} → {attrs['pooling']} → {stored} values"
                             if "pooling" in attrs and len(attrs["output_shape"]) > 1 else f"{stored} values")
        unique = list(dict.fromkeys(parts))
        return unique[0] if len(parts) == 1 else (f"{len(parts)} calls, each {unique[0]}" if len(unique) == 1
                                                  else "; ".join(parts))

    def layer_nbytes(self, name):
        self._check_layer(name)
        with h5py.File(self.h5_path, "r") as f:
            return sum(f[call].size * f[call].dtype.itemsize for call in self.calls_of(name))

    def _check_layer(self, name):
        if name not in self.layer_names:
            raise KeyError(f"Layer '{name}' not found in cache.")


def _failed_rows(output_dir, attrs, metadata, config, warnings):
    """(failed rows or None, rows attempted, note or None) for a cache's run. The note goes first
    in `warnings`: every analysis of these results is missing those rows."""
    n_failed = int(attrs.get("rows_failed", 0))
    rows_attempted = int(attrs.get("rows_attempted", len(metadata) + n_failed))
    if not n_failed:
        return None, rows_attempted, None
    head = (f"{n_failed} of {rows_attempted} rows ({percent(n_failed, rows_attempted)}) failed during the "
            "extraction and are not in these results")
    path = os.path.join(output_dir, FAILED_NAME)
    failures = None
    if os.path.exists(path):
        failures = read_metadata(path, config.id_col if config else None)
        if len(failures) != n_failed or not {FAILED_AT, ERROR} <= set(failures.columns):
            failures = None
    if failures is None:
        note = f"{head}. {FAILED_NAME}, which should list them, is missing or from another run."
    else:
        exclude = [c for c in (config.id_col, config.group_col) if c] if config else []
        attempted = pd.concat([metadata, failures.drop(columns=[FAILED_AT, ERROR])], ignore_index=True)
        stages, groups = describe(attempted, [False] * len(metadata) + [True] * n_failed, failures, exclude)
        note = (f"{head}: {stages}." + (f" Most affected: {groups}." if groups else "")
                + f" {FAILED_NAME} lists each one with its error.")
    warnings.insert(0, note)
    return failures, rows_attempted, note


def _calls_by_module(layer_names):
    """{module: [its "<module>@<i>" datasets, in call order]} for modules stored once per call."""
    calls = {}
    for name in layer_names:
        module, _, call = name.rpartition("@")
        if module and call.isdigit():
            calls.setdefault(module, []).append(name)
    return {m: sorted(names, key=lambda n: int(n.rpartition("@")[2])) for m, names in calls.items()}


class _SideBySide:
    """Several calls of one module read as one wider layer, row-sliced like an h5py dataset."""

    def __init__(self, datasets):
        self.datasets = datasets
        widths = [int(np.prod(d.shape[1:])) for d in datasets]
        self.shape = (datasets[0].shape[0], sum(widths))
        self.dtype = np.result_type(*[d.dtype for d in datasets])

    def __getitem__(self, rows):
        blocks = [np.asarray(d[rows]) for d in self.datasets]
        return np.hstack([b.reshape(len(b), -1) for b in blocks])
