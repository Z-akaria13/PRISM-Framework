"""core.config.AuditConfig: the job description shared by the GUI, the dispatcher and headless.py."""
import json

import pytest

from prism.core.config import DEFAULT_POOLING, POOLING_STRATEGIES, AuditConfig, parse_layer_label

REQUIRED = dict(project_root="/proj", script_path="/proj/connector.py", framework="PyTorch",
                csv_path="/proj/meta.csv", id_col="subject_id", target_col="diagnosis", output_dir="/out")


def make(**overrides):
    return AuditConfig(**{**REQUIRED, **overrides})


def test_defaults():
    config = make()

    assert (config.group_col, config.layers, config.batch_size, config.device) == (None, {}, 16, "auto")


def test_json_round_trip_preserves_everything_including_layer_order():
    config = make(group_col="patient", layers={"layer4": "Max Pool", "fc": "Mean Pool", "conv1": "Flatten"},
                  batch_size=8, device="cuda:1")

    restored = AuditConfig.from_json(config.to_json())

    assert restored == config
    assert list(restored.layers) == ["layer4", "fc", "conv1"]


def test_json_uses_the_job_file_keys_headless_has_always_read():
    data = json.loads(make().to_json())

    assert set(data) == {"project_root", "script_path", "framework", "csv_path", "id_col", "target_col",
                         "group_col", "layers", "batch_size", "output_dir", "device", "overwrite",
                         "max_failed_percent"}


def test_legacy_job_files_with_layer_labels_are_read():
    legacy = {**REQUIRED, "group_col": None, "batch_size": 4,
              "layers": ["layer1.0.conv1 [Max Pool]", "fc", "encoder.0 [CLS Token]"]}

    config = AuditConfig.from_dict(legacy)

    assert config.layers == {"layer1.0.conv1": "Max Pool", "fc": DEFAULT_POOLING, "encoder.0": "CLS Token"}
    assert config.device == "auto"


@pytest.mark.parametrize("label, expected", [
    ("conv", ("conv", "Mean Pool")),
    ("conv [Flatten]", ("conv", "Flatten")),
    ("blocks.3.attn [Seq Pool]", ("blocks.3.attn", "Seq Pool")),
])
def test_parse_layer_label(label, expected):
    assert parse_layer_label(label) == expected


def test_unknown_keys_are_ignored():
    config = AuditConfig.from_dict({**REQUIRED, "written_by": "PRISM 1.1"})

    assert config == make()


def test_missing_required_keys_are_named():
    data = {k: v for k, v in REQUIRED.items() if k not in ("csv_path", "target_col")}

    with pytest.raises(ValueError, match=r"\['csv_path', 'target_col'\]"):
        AuditConfig.from_dict(data)


@pytest.mark.parametrize("batch_size", [0, -4, "16", 2.5, True])
def test_batch_size_must_be_a_positive_integer(batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        make(batch_size=batch_size)


def test_unknown_pooling_strategy_is_rejected():
    with pytest.raises(ValueError, match="Unknown pooling strategy"):
        make(layers={"conv": "Average"})


@pytest.mark.parametrize("strategy", POOLING_STRATEGIES)
def test_every_wiretap_strategy_is_accepted(strategy):
    assert make(layers={"conv": strategy}).layers == {"conv": strategy}


@pytest.mark.parametrize("device", ["auto", "cpu", "cuda", "cuda:0", "cuda:3"])
def test_valid_devices(device):
    assert make(device=device).device == device


def test_unknown_device_is_rejected():
    with pytest.raises(ValueError, match="Unknown device"):
        make(device="tpu")


def test_config_is_immutable_and_owns_its_layers():
    layers = {"conv": "Max Pool"}
    config = make(layers=layers)
    layers["fc"] = "Mean Pool"

    assert config.layers == {"conv": "Max Pool"}
    with pytest.raises(AttributeError):
        config.batch_size = 32


def test_overwrite_defaults_off_and_round_trips():
    assert make().overwrite is False
    assert AuditConfig.from_json(make(overwrite=True).to_json()).overwrite is True


def test_overwrite_must_be_a_boolean():
    with pytest.raises(ValueError, match="overwrite must be true or false"):
        make(overwrite="yes")


def test_the_failure_limit_defaults_to_5_percent_and_round_trips():
    assert make().max_failed_percent == 5.0
    assert AuditConfig.from_json(make(max_failed_percent=12.5).to_json()).max_failed_percent == 12.5


def test_job_files_written_before_the_failure_limit_get_the_default():
    data = json.loads(make().to_json())
    del data["max_failed_percent"]

    assert AuditConfig.from_dict(data).max_failed_percent == 5.0


@pytest.mark.parametrize("limit", [-1, 100.5, "5", True, float("nan")])
def test_the_failure_limit_must_be_a_percentage(limit):
    with pytest.raises(ValueError, match="max_failed_percent must be a number from 0 to 100"):
        make(max_failed_percent=limit)



def test_grid_pool_is_a_pooling_strategy():
    assert make(layers={"conv": "Grid Pool"}).layers == {"conv": "Grid Pool"}
