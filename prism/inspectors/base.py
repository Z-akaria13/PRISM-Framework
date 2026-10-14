"""
THE INSPECTOR INTERFACE
An inspector wraps one framework's model: it lists the model's layers for the Wiretap tab and
captures the chosen layers' activations while the connector runs a batch. Subclass Inspector
for a new framework and register it (see prism/inspectors/__init__.py).

The extraction engine calls, per run:
    attach_hooks(layer names) -> for each batch: run_inference, retrieve_features(rows), clear_buffer
    -> teardown()
and, if the inspector defines them:
    to_device(device) -> the device used         (at run start)
    prepare_for_inference() -> warning or None   (at run start, e.g. eval mode)
    inference_context() -> context manager       (around each run_inference, e.g. no_grad)
The Wiretap's Show shapes calls output_shapes(layer names, run) on one item, and each run records
model_fingerprint() with its results.
"""
from typing import Any, Dict, List, Optional

import numpy as np


class Inspector:
    #: False for black-box frameworks: no layers to choose, only the model's output is cached.
    extracts_hidden_layers = True

    def __init__(self, model: Any):
        self.model = model
        self._buffer: Dict[str, List[np.ndarray]] = {}

    def discover_topology(self) -> List[Dict[str, Any]]:
        """The layers that can be captured: [{"id": name, "type": class name, "params": count}]."""
        return []

    def attach_hooks(self, layer_names: List[str]):
        """Start capturing these layers' outputs into the buffer."""

    def retrieve_features(self, expected_rows: Optional[int] = None) -> Dict[str, np.ndarray]:
        """The activations captured since the buffer was cleared: {layer: rows x ...}, rows in input
        order. The default joins whatever `_buffer` holds per layer along the rows."""
        return {name: arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=0)
                for name, arrays in self._buffer.items() if arrays}

    def model_fingerprint(self) -> Optional[str]:
        """'sha256:<hex>' of the model's weights as loaded, recorded with the results; None if this
        framework can't tell."""
        return None

    def output_shapes(self, layer_names: List[str], run) -> Dict[str, tuple]:
        """{dataset name: shape of one row's output} for these layers while `run()` runs the model on
        one item, named as retrieve_features names them. The default captures the activations;
        an inspector can record only their shapes."""
        self.attach_hooks(layer_names)
        try:
            run()
            return {name: tuple(np.shape(a)[1:]) for name, a in self.retrieve_features(expected_rows=1).items()}
        finally:
            self.clear_buffer()

    def clear_buffer(self):
        self._buffer = {}

    def teardown(self):
        """Leave the user's model as it was before the run."""
        self.clear_buffer()
