"""core.labels: the single decoding of model outputs shared by the analysis tabs."""
import numpy as np
import pytest

from prism.core.labels import class_order, decode_predictions


def test_declared_class_names_set_the_order():
    order, notes = class_order(["AD", "CN", "MCI", "CN"], class_names=["CN", "MCI", "AD"])

    assert order == ["CN", "MCI", "AD"]
    assert notes == []


def test_integer_labels_need_no_declaration():
    order, notes = class_order([0, 2, 1, 1])

    assert order == ["0", "1", "2"]
    assert notes == []


def test_undeclared_string_labels_fall_back_to_alphabetical_order_with_a_note():
    order, notes = class_order(["CN", "AD", "MCI"])

    assert order == ["AD", "CN", "MCI"]
    assert len(notes) == 1 and "class_names" in notes[0]


def test_labels_missing_from_class_names_are_noted():
    _, notes = class_order(["CN", "AD", "LMCI"], class_names=["CN", "AD"])

    assert "LMCI" in notes[0]


def test_missing_labels_are_ignored():
    order, _ = class_order(["b", None, "a", np.nan])

    assert order == ["a", "b"]


def test_multi_column_outputs_take_the_highest_scoring_class():
    raw = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8], [0.2, 0.5, 0.3]])

    assert list(decode_predictions(raw, ["CN", "MCI", "AD"])) == ["CN", "AD", "MCI"]


@pytest.mark.parametrize("raw", [np.array([0.5, 0.49, 0.9]), np.array([[0.5], [0.49], [0.9]])])
def test_single_scores_are_thresholded_at_one_half_inclusive(raw):
    assert list(decode_predictions(raw, ["neg", "pos"])) == ["pos", "neg", "pos"]


def test_logits_are_thresholded_at_zero():
    assert list(decode_predictions(np.array([0.0, -0.1, 2.0]), ["neg", "pos"], "logits")) == ["pos", "neg", "pos"]


def test_output_width_must_match_the_number_of_classes():
    with pytest.raises(ValueError, match="class_names"):
        decode_predictions(np.zeros((2, 3)), ["0", "1"])


def test_unknown_output_kind_is_rejected():
    with pytest.raises(ValueError, match="output_kind"):
        decode_predictions(np.zeros(2), ["0", "1"], "percent")
