"""
CONNECTOR LOADING
Imports the user's connector script and instantiates the connector class defined in it.
"""
import importlib.abc
import importlib.machinery
import importlib.util
import inspect
import os
import sys

import prism.core.connectorTemplate as _template_camel
import prism.core.connector_template as _template_snake
from prism.connector import PrismConnector

# Connectors written for older versions import the template as a top-level `core` package
# (`from core.connectorTemplate import WhydunnitConnector`), from when PRISM ran with its own
# folder on sys.path. Those names resolve to the one real module, so the class is the same object.
_LEGACY_MODULES = {"core.connectorTemplate": _template_camel, "core.connector_template": _template_snake}


class _LegacyTemplateFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Serves `core.connectorTemplate` / `core.connector_template`, and an empty `core` package to
    hold them only when the user's project has no `core` package of its own."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname in _LEGACY_MODULES:
            return importlib.machinery.ModuleSpec(fullname, self)
        if fullname == "core" and importlib.machinery.PathFinder.find_spec("core", path) is None:
            spec = importlib.machinery.ModuleSpec("core", self, is_package=True)
            spec.submodule_search_locations = []
            return spec
        return None

    def create_module(self, spec):
        return _LEGACY_MODULES.get(spec.name)  # None: a fresh, empty `core` package

    def exec_module(self, module):
        if module.__name__ == "core":
            module.__prism_placeholder__ = True


def _forget_placeholder_core():
    """Drop the empty `core` package from the module cache, so a project that has a `core`
    package of its own (loaded later in the same session) can still import it."""
    if getattr(sys.modules.get("core"), "__prism_placeholder__", False):
        del sys.modules["core"]


def _install_legacy_names():
    if not any(isinstance(finder, _LegacyTemplateFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _LegacyTemplateFinder())


_install_legacy_names()


def load_user_connector(script_path: str, project_root: str) -> PrismConnector:
    """Import `script_path` (with `project_root` importable) and return its connector, constructed."""
    # 1. Add user's project root to path so their internal imports work
    if project_root and project_root not in sys.path:
        sys.path.insert(0, project_root)

    # 2. Load the module dynamically
    module_name = os.path.basename(script_path).replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if not spec or not spec.loader:
        raise ImportError(f"Could not load script: {script_path}")
    user_module = importlib.util.module_from_spec(spec)
    _forget_placeholder_core()
    try:
        spec.loader.exec_module(user_module)
    except Exception as e:
        raise ImportError(f"Error executing user script: {e}")
    finally:
        _forget_placeholder_core()

    # 3. The connector class it defines
    return _connector_class(user_module)(project_root)


def _connector_class(module):
    """The concrete PrismConnector subclass in the script: the most derived one if the script builds
    one connector on another, preferring classes defined in the script over imported ones."""
    found = [obj for _, obj in inspect.getmembers(module, inspect.isclass)
             if issubclass(obj, PrismConnector) and obj is not PrismConnector]
    concrete = [cls for cls in found if not inspect.isabstract(cls)]
    if not concrete:
        if found:
            cls = found[-1]
            missing = ", ".join(sorted(cls.__abstractmethods__))
            raise ValueError(f"{cls.__name__} does not implement {missing}, which every connector needs.")
        stray = [obj.__name__ for _, obj in inspect.getmembers(module, inspect.isclass)
                 if any(base.__name__ in ("PrismConnector", "WhydunnitConnector") for base in obj.__mro__[1:])]
        if stray:
            raise ValueError(f"{', '.join(stray)} is built on a different copy of the PRISM connector template. "
                             "Import it with `from prism import PrismConnector`.")
        raise ValueError("No class inheriting from PrismConnector found in script.")

    leaves = [cls for cls in concrete if not any(other is not cls and issubclass(other, cls) for other in concrete)]
    local = [cls for cls in leaves if cls.__module__ == module.__name__]
    choices = local or leaves
    if len(choices) > 1:
        names = ", ".join(cls.__name__ for cls in choices)
        raise ValueError(f"The script defines several connectors ({names}); keep one, or build one on the others.")
    return choices[0]
