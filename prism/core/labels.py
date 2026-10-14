"""
PREDICTION DECODING
Turns raw model outputs into target labels. Shared by the Performance and
Error Gallery tabs so they can never disagree about what the model predicted.
"""
import numpy as np
import pandas as pd

OUTPUT_KINDS = (None, "probabilities", "logits")


def class_order(labels, class_names=None):
    """The target labels (as strings) in model-output order, plus notes on how that order was decided.

    Column j of a multi-column output is class_order[j]; a single-column output scores
    the last class against the first.
    """
    observed = sorted({str(v) for v in pd.Series(labels).dropna()})
    if class_names is not None:
        order = [str(c) for c in class_names]
        unknown = sorted(set(observed) - set(order))
        notes = [f"Labels {unknown} are not in the connector's class_names {order}."] if unknown else []
        return order, notes
    if observed == [str(i) for i in range(len(observed))]:
        return observed, []  # labels 0..k-1: output column j is class j
    return observed, [f"The connector does not declare class_names, so the model's outputs are assumed to be "
                      f"in alphabetical label order {observed}. Set class_names on the connector if that is wrong."]


def decode_predictions(raw, order, output_kind=None):
    """Predicted label (string) for every row of `raw`."""
    if output_kind not in OUTPUT_KINDS:
        raise ValueError(f"Unknown output_kind {output_kind!r}; expected 'probabilities' or 'logits'")
    raw = np.asarray(raw)
    if raw.ndim > 1 and raw.shape[1] > 1:
        if raw.shape[1] != len(order):
            raise ValueError(f"The model outputs {raw.shape[1]} scores per item but there are {len(order)} "
                             f"classes {order}. Set class_names on the connector to the model's full class list.")
        return np.asarray(order)[raw.argmax(axis=1)]
    threshold = 0.0 if output_kind == "logits" else 0.5
    return np.where(raw.reshape(len(raw)) >= threshold, order[-1], order[0])
