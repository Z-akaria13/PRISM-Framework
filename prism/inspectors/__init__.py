"""
INSPECTOR REGISTRY
The framework plugins, by the name the Setup tab and job files use. Each is imported the first
time it is used, so PyTorch or TensorFlow are needed only for audits of models built with them.

Another package can add an inspector (a subclass of prism.inspectors.base.Inspector) through the
"prism.inspectors" entry point group, in its pyproject.toml:

    [project.entry-points."prism.inspectors"]
    "JAX" = "my_package.jax_inspector:JaxInspector"
"""
import importlib
from importlib import metadata

from prism.inspectors.base import Inspector  # noqa: F401  (the interface plugins implement)

BUILT_IN = {
    "PyTorch": "prism.inspectors.pytorch:PyTorchInspector",
    "TensorFlow/Keras": "prism.inspectors.keras:KerasInspector",
    "Scikit-Learn": "prism.inspectors.sklearn:SklearnInspector",
}
ENTRY_POINT_GROUP = "prism.inspectors"
_INSTALL_HINTS = {"torch": "pip install torch", "tensorflow": "pip install tensorflow"}


class InspectorRef:
    """A plugin by name and import path, imported on first use. Calling it builds the inspector."""

    def __init__(self, name, target):
        self.name, self.target = name, target

    def load(self):
        module_name, _, attr = self.target.partition(":")
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            missing = (e.name or "").split(".")[0]
            hint = _INSTALL_HINTS.get(missing, f"pip install {missing}" if missing else "")
            raise ImportError(f"The {self.name} inspector needs '{missing or module_name}', which is not "
                              f"installed{': ' + hint if hint else ''}.") from e
        return getattr(module, attr)

    def __call__(self, model):
        return self.load()(model)

    def __repr__(self):
        return f"InspectorRef({self.name!r}, {self.target!r})"


def get_inspectors():
    """{display name: InspectorRef} for the built-in plugins and any installed through entry points."""
    registry = {name: InspectorRef(name, target) for name, target in BUILT_IN.items()}
    try:
        entry_points = metadata.entry_points(group=ENTRY_POINT_GROUP)
    except Exception as e:  # noqa: BLE001 - a broken installation must not stop the built-ins
        print(f"[WARNING] Could not read installed inspector plugins: {e}")
        return registry
    for entry_point in entry_points:
        if entry_point.name in BUILT_IN:
            print(f"[WARNING] Ignoring the plugin '{entry_point.value}': '{entry_point.name}' is a built-in inspector.")
            continue
        registry[entry_point.name] = InspectorRef(entry_point.name, entry_point.value)
    return registry
