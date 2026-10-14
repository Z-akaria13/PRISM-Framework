"""
PROVENANCE
What produced a set of results, recorded with them so a run can be traced and compared later:
the software and its versions, when it ran, and fingerprints of its inputs (the metadata CSV,
the connector script, and the model's weights as loaded, wherever the connector found them).
"""
import datetime
import hashlib
import os
import platform
from importlib import metadata

import numpy as np

from prism import __version__

PACKAGES = ("numpy", "pandas", "scikit-learn", "scipy", "h5py", "torch", "torchvision", "tensorflow")


def now():
    """The local time with its UTC offset, e.g. '2026-10-14T10:20:31+01:00'."""
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def environment():
    """PRISM's version, Python's, the platform, and the installed versions of the packages that
    shape the numbers."""
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return {"prism": __version__, "python": platform.python_version(), "platform": platform.platform(),
            "packages": packages}


def file_digest(path):
    """'sha256:<hex>' of a file's bytes (what `sha256sum` prints), or None if there is no such file."""
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def weights_digest(entries):
    """'sha256:<hex>' of named arrays such as a model's weights: (name, dtype, shape, raw bytes) per
    array, combined in name order, so the order they are listed in doesn't matter."""
    digest = hashlib.sha256()
    for name, dtype, shape, data in sorted(entries, key=lambda entry: entry[0]):
        for part in (name, str(dtype), str(tuple(shape))):
            digest.update(part.encode() + b"\0")
        digest.update(data)
    return "sha256:" + digest.hexdigest()


def array_entries(named_arrays):
    """weights_digest entries for (name, numpy-convertible array) pairs."""
    for name, array in named_arrays:
        array = np.ascontiguousarray(array)
        yield name, array.dtype, array.shape, array.tobytes()


def inputs(config, inspector, log_callback):
    """Fingerprints of a run's inputs, taken as it starts. A model that can't be fingerprinted is
    logged and recorded as None; it doesn't stop the run."""
    model = None
    fingerprint = getattr(inspector, "model_fingerprint", None)
    if fingerprint is not None:
        try:
            model = fingerprint()
        except Exception as e:  # noqa: BLE001 - a fingerprint is a record, not a requirement
            log_callback(f"[WARNING] Could not fingerprint the model's weights ({type(e).__name__}: {e}).")
    if model:
        log_callback(f"Model fingerprint: {model}")
    return {"metadata_csv": {"path": config.csv_path, "sha256": file_digest(config.csv_path)},
            "connector": {"path": config.script_path, "sha256": file_digest(config.script_path)},
            "model": model}
