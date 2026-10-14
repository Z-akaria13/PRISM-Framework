"""inspectors/: plugin discovery and the PyTorch, Keras and scikit-learn inspectors."""
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from _support import known_bug
from prism.inspectors import Inspector, get_inspectors
from prism.inspectors.keras import KerasInspector
from prism.inspectors.sklearn import SklearnInspector

torch = pytest.importorskip("torch")
nn = torch.nn
from prism.inspectors.pytorch import PyTorchInspector  # noqa: E402  (needs torch)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_lists_the_bundled_plugins():
    registry = get_inspectors()

    assert {name: ref.load() for name, ref in registry.items()} == {
        "PyTorch": PyTorchInspector,
        "TensorFlow/Keras": KerasInspector,
        "Scikit-Learn": SklearnInspector,
    }
    assert all(issubclass(ref.load(), Inspector) for ref in registry.values())


def test_listing_the_frameworks_does_not_import_them():
    import subprocess
    from _support import REPO_ROOT
    code = ("import sys; from prism.inspectors import get_inspectors; names = list(get_inspectors()); "
            "print(names, 'torch' in sys.modules, 'tensorflow' in sys.modules)")

    out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=True)

    assert out.stdout.strip() == "['PyTorch', 'TensorFlow/Keras', 'Scikit-Learn'] False False"


def test_a_framework_that_is_not_installed_is_named_with_how_to_install_it(monkeypatch):
    from prism.inspectors import InspectorRef
    monkeypatch.setitem(sys.modules, "torch", None)  # makes `import torch` fail
    monkeypatch.delitem(sys.modules, "prism.inspectors.pytorch", raising=False)

    with pytest.raises(ImportError, match="needs 'torch', which is not installed: pip install torch"):
        InspectorRef("PyTorch", "prism.inspectors.pytorch:PyTorchInspector")(object())


def test_installed_packages_can_add_inspectors(monkeypatch):
    from importlib import metadata
    import prism.inspectors as registry_module
    plugin = types.ModuleType("prism_test_jax_plugin")

    class JaxInspector(Inspector):
        pass

    plugin.JaxInspector = JaxInspector
    monkeypatch.setitem(sys.modules, "prism_test_jax_plugin", plugin)
    points = [metadata.EntryPoint("JAX", "prism_test_jax_plugin:JaxInspector", "prism.inspectors"),
              metadata.EntryPoint("PyTorch", "prism_test_jax_plugin:JaxInspector", "prism.inspectors")]
    monkeypatch.setattr(registry_module.metadata, "entry_points", lambda group: points)

    registry = get_inspectors()

    assert isinstance(registry["JAX"](object()), JaxInspector)
    assert registry["PyTorch"].load() is PyTorchInspector  # a plugin cannot replace a built-in


# ---------------------------------------------------------------------------
# Scikit-learn (black box)
# ---------------------------------------------------------------------------

def test_sklearn_inspector_is_a_no_op_black_box():
    inspector = SklearnInspector(object())

    assert inspector.extracts_hidden_layers is False
    assert inspector.discover_topology() == []
    inspector.attach_hooks(["anything"])
    assert inspector.retrieve_features() == {}
    inspector.clear_buffer()
    inspector.teardown()


# ---------------------------------------------------------------------------
# PyTorch
# ---------------------------------------------------------------------------

def mlp():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2)).eval()


def test_discover_topology_lists_leaf_modules_with_param_counts():
    model = nn.Sequential(nn.Linear(4, 8), nn.Sequential(nn.ReLU(), nn.Linear(8, 1000)))

    layers = PyTorchInspector(model).discover_topology()

    assert layers == [
        {"id": "0", "type": "Linear", "params": "40"},
        {"id": "1.0", "type": "ReLU", "params": "0"},
        {"id": "1.1", "type": "Linear", "params": "9,000"},
    ]


def test_hooks_capture_batch_first_numpy_activations():
    model = mlp()
    inspector = PyTorchInspector(model)
    inspector.attach_hooks(["0", "2"])
    x = torch.randn(5, 4)

    with torch.no_grad():
        out = model(x)
    features = inspector.retrieve_features()

    assert set(features) == {"0", "2"}
    assert isinstance(features["0"], np.ndarray)
    assert features["0"].shape == (5, 3)
    np.testing.assert_allclose(features["2"], out.numpy(), rtol=1e-6)


def test_tuple_outputs_keep_the_first_element():
    class Recurrent(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(3, 4, batch_first=True)

        def forward(self, x):
            return self.lstm(x)[0]

    model = Recurrent().eval()
    inspector = PyTorchInspector(model)
    inspector.attach_hooks(["lstm"])

    with torch.no_grad():
        model(torch.randn(2, 5, 3))

    assert inspector.retrieve_features()["lstm"].shape == (2, 5, 4)


def test_non_tensor_outputs_are_ignored():
    class DictOut(nn.Module):
        def forward(self, x):
            return {"y": x}

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = DictOut()

        def forward(self, x):
            return self.inner(x)["y"]

    model = Wrapper()
    inspector = PyTorchInspector(model)
    inspector.attach_hooks(["inner"])

    model(torch.randn(2, 3))

    assert inspector.retrieve_features() == {}


def test_clear_buffer_keeps_hooks_and_teardown_removes_them():
    model = mlp()
    inspector = PyTorchInspector(model)
    inspector.attach_hooks(["0"])
    x = torch.randn(2, 4)

    model(x)
    inspector.clear_buffer()
    assert inspector.retrieve_features() == {}

    model(x)
    assert inspector.retrieve_features()["0"].shape == (2, 3)

    inspector.teardown()
    assert len(model[0]._forward_hooks) == 0
    model(x)
    assert inspector.retrieve_features() == {}


def test_reattaching_hooks_does_not_double_register():
    model = mlp()
    inspector = PyTorchInspector(model)
    inspector.attach_hooks(["0"])
    inspector.attach_hooks(["0"])

    model(torch.randn(3, 4))

    assert inspector.retrieve_features()["0"].shape == (3, 3)


def test_module_called_twice_returns_one_array_per_call():
    """A module run twice in one forward pass (e.g. a ResNet block's shared ReLU)."""
    class Reused(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(3, 3)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.lin(self.relu(x)))

    model = Reused().eval()
    inspector = PyTorchInspector(model)
    inspector.attach_hooks(["relu"])

    with torch.no_grad():
        model(torch.randn(4, 3))

    features = inspector.retrieve_features(expected_rows=4)
    assert set(features) == {"relu@0", "relu@1"}
    assert features["relu@0"].shape == (4, 3) and features["relu@1"].shape == (4, 3)


def test_parameterless_model_runs_through_the_device_hook():
    model = nn.Sequential(nn.ReLU())
    inspector = PyTorchInspector(model)
    inspector.attach_hooks(["0"])

    model(torch.randn(2, 3))

    assert inspector.retrieve_features()["0"].shape == (2, 3)


def test_device_pre_hook_moves_nested_inputs_to_the_model_device():
    class TakesNested(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(2, 2)

        def forward(self, batch):
            return self.lin(batch["x"]) + self.lin(batch["pair"][0])

    model = TakesNested().to("meta")
    PyTorchInspector(model).attach_hooks([])

    out = model({"x": torch.ones(1, 2), "pair": (torch.ones(1, 2),)})

    assert out.device.type == "meta"


@pytest.fixture
def recorded_moves(monkeypatch):
    model = nn.Linear(2, 2)
    moves = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(model, "to", lambda device: moves.append(device) or model)
    return model, moves


def test_creating_an_inspector_leaves_the_model_where_it_is(recorded_moves):
    """Setup only lists layers; the GPU is claimed when a run starts, on the machine running it."""
    model, moves = recorded_moves

    PyTorchInspector(model)

    assert moves == []


@pytest.mark.parametrize("requested, expected", [("auto", "cuda"), ("cpu", "cpu"), ("cuda:1", "cuda:1")])
def test_to_device_resolves_and_applies_the_device(recorded_moves, requested, expected):
    model, moves = recorded_moves

    used = PyTorchInspector(model).to_device(requested)

    assert used == expected and moves == [expected]


def test_auto_device_falls_back_to_cpu_without_cuda():
    assert PyTorchInspector(nn.Linear(2, 2)).to_device("auto") == "cpu"


def test_teardown_leaves_the_users_model_unmodified():
    model = mlp()
    pre_hooks_before = len(model._forward_pre_hooks)
    inspector = PyTorchInspector(model)
    inspector.attach_hooks(["0"])

    inspector.teardown()

    assert len(model._forward_pre_hooks) == pre_hooks_before


# ---------------------------------------------------------------------------
# Keras (driven through a fake `tensorflow` module, so TF is not required)
# ---------------------------------------------------------------------------

class FakeTensor:
    def __init__(self, values):
        self._values = np.asarray(values)

    def numpy(self):
        return self._values


class FakeLayer:
    def __init__(self, name, fn, n_params=0):
        self.name = name
        self.fn = fn
        self._n_params = n_params
        self.output = None  # set by FakeKerasModel

    def count_params(self):
        if self._n_params is None:
            raise ValueError("layer has not been built")
        return self._n_params


class FakeKerasModel:
    """Duck-types what KerasInspector touches. `__call__` lives on the class, as in real Keras."""

    def __init__(self, layers):
        self.layers = layers
        self.inputs = ["<input>"]
        for layer in layers:
            layer.output = (self, layer.name)

    def get_layer(self, name):
        return next(layer for layer in self.layers if layer.name == name)

    def activations(self, x):
        acts = {}
        for layer in self.layers:
            x = layer.fn(x)
            acts[layer.name] = x
        return acts

    def __call__(self, x, *args, **kwargs):
        return self.activations(x)[self.layers[-1].name]

    def predict(self, x, verbose=0):
        return self(x)


class FakeFunctionalModel:
    def __init__(self, inputs, outputs):
        self._outputs = outputs

    def __call__(self, x, training=False):
        results = [FakeTensor(model.activations(x)[name]) for model, name in self._outputs]
        return results if len(results) > 1 else results[0]


@pytest.fixture
def fake_tf(monkeypatch):
    module = types.ModuleType("tensorflow")
    module.keras = SimpleNamespace(Model=FakeFunctionalModel)
    monkeypatch.setitem(sys.modules, "tensorflow", module)
    return module


def keras_cnn():
    """(B, H, W, C) image -> channels-last conv feature map -> dense head."""
    return FakeKerasModel([
        FakeLayer("conv", lambda x: np.repeat(x, 3, axis=-1) * np.array([1.0, 2.0, 3.0]), n_params=30),
        FakeLayer("dense", lambda x: x.mean(axis=(1, 2)), n_params=None),
    ])


def test_keras_topology_lists_layers_and_tolerates_unbuilt_ones():
    layers = KerasInspector(keras_cnn()).discover_topology()

    assert [layer["id"] for layer in layers] == ["conv", "dense"]
    assert layers[0]["params"] == "30"
    assert layers[1]["params"] == "0"


def test_keras_hook_transposes_channels_last_maps_to_channels_first(fake_tf):
    model = keras_cnn()
    inspector = KerasInspector(model)
    inspector.attach_hooks(["conv", "dense"])
    x = np.random.default_rng(0).normal(size=(2, 4, 5, 1))

    model.__call__(x)  # explicit attribute call: the only path the current hook intercepts
    features = inspector.retrieve_features()

    assert features["conv"].shape == (2, 3, 4, 5)
    assert features["dense"].shape == (2, 3)


def test_keras_teardown_stops_capturing(fake_tf):
    model = keras_cnn()
    inspector = KerasInspector(model)
    inspector.attach_hooks(["conv"])

    inspector.teardown()
    model.__call__(np.zeros((1, 2, 2, 1)))

    assert inspector.retrieve_features() == {}


@known_bug("P0 #5")
@pytest.mark.parametrize("call_style", ["model(x)", "model.predict(x)"])
def test_keras_hook_fires_on_normal_calls(fake_tf, call_style):
    model = keras_cnn()
    inspector = KerasInspector(model)
    inspector.attach_hooks(["conv"])
    x = np.zeros((2, 4, 4, 1))

    model(x) if call_style == "model(x)" else model.predict(x)

    assert "conv" in inspector.retrieve_features()


def test_eval_mode_is_set_for_the_run_and_a_mix_of_modes_restored_exactly():
    model = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.5)).train()
    model[1].eval()
    inspector = PyTorchInspector(model)

    warning = inspector.prepare_for_inference()
    inspector.attach_hooks(["0"])  # replacing hooks must not undo the switch

    assert "training mode (2 modules)" in warning
    assert not any(module.training for module in model.modules())
    inspector.teardown()
    assert model.training and model[0].training and not model[1].training


def test_inference_context_disables_autograd():
    with PyTorchInspector(nn.Linear(2, 2)).inference_context():
        assert not torch.is_grad_enabled()



class TwoStage(nn.Module):
    """A model a connector might drive one submodule at a time, never calling the whole model."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(3, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.backbone(self.relu(x)))


def test_calls_from_direct_submodule_use_are_told_apart_by_row_counts():
    model = TwoStage().eval()
    inspector = PyTorchInspector(model)
    inspector.attach_hooks(["backbone", "relu"])

    with torch.no_grad():
        for item in torch.randn(3, 1, 3):   # one item at a time, through submodules only
            model.relu(model.backbone(item))

    features = inspector.retrieve_features(expected_rows=3)
    assert features["backbone"].shape == (3, 3) and features["relu"].shape == (3, 3)


def test_a_layer_called_a_different_number_of_times_per_pass_is_refused():
    class Variable(nn.Module):
        def __init__(self):
            super().__init__()
            self.cell = nn.Linear(2, 2)

        def forward(self, x, steps):
            for _ in range(steps):
                x = self.cell(x)
            return x

    model = Variable()
    inspector = PyTorchInspector(model)
    inspector.attach_hooks(["cell"])
    with torch.no_grad():
        model(torch.randn(1, 2), 2)
        model(torch.randn(1, 2), 3)

    with pytest.raises(ValueError, match="ran a different number of times in different forward passes"):
        inspector.retrieve_features(expected_rows=2)
