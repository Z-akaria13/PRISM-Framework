"""
THE CONNECTOR
Subclass PrismConnector in a script of your own to bridge your model and data into PRISM:

    from prism import PrismConnector

    class MyConnector(PrismConnector):
        def load_model(self): ...
        def load_item(self, row_id, metadata_row): ...
        def run_inference(self, model, batch_data): ...

Point the Setup tab (or a job file) at that script. PRISM picks the connector class defined
in it, the most derived one if there are several.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class PrismConnector(ABC):
    """The user protocol: how PRISM loads your model, reads one item and runs a batch."""

    #: Optional. The target-column values in the order of your model's output columns,
    #: e.g. ["CN", "MCI", "AD"]; for a single-column output, [negative, positive].
    #: Without it, outputs are assumed to follow alphabetical label order and the
    #: analysis tabs say so.
    class_names = None

    #: Optional. What a single-column output holds: "probabilities" (thresholded at 0.5,
    #: the default) or "logits" (thresholded at 0).
    output_kind = None

    def __init__(self, project_root: str):
        self.project_root = project_root

    @abstractmethod
    def load_model(self) -> Any:
        """
        Return the raw model object.
        - PyTorch: return nn.Module
        - Sklearn: return estimator
        """

    @abstractmethod
    def load_item(self, row_id: str, metadata_row: Dict) -> Any:
        """
        Return the raw input data for a single row.

        Args:
            row_id: The row's value in the ID column chosen in the Evidence tab, as a string.
            metadata_row: The whole metadata row, as a dict.
        """

    @abstractmethod
    def run_inference(self, model: Any, batch_data: List[Any]) -> Any:
        """
        Run the model on a batch: `batch_data` is a list of items from load_item, as many as
        the batch size set for the run. PRISM's hooks fire during this call.

        If you can't run true hardware batches, loop over `batch_data` and stack the outputs.
        PyTorch RNNs and LSTMs need batch_first=True: PRISM expects the batch dimension first
        in every hidden representation.
        """

    def render_item_for_gallery(self, raw_data_item) -> dict:
        """
        Optional: how the Error Gallery displays one item. The default shows str(item).

        Args:
            raw_data_item: The exact object returned by your `load_item()` function.

        Returns:
            A dictionary with the rendering type and its payload:
            - {'type': 'image', 'payload': array}  (H x W grayscale, or C x H x W colour)
            - {'type': 'text',  'payload': "Patient presented with..."}
            - {'type': 'table', 'payload': {'Age': 45, 'Income': 50000}}
        """
        return {'type': 'text', 'payload': str(raw_data_item)}


#: The connector's name before PRISM was renamed; the same class, so older connectors keep working.
WhydunnitConnector = PrismConnector
