"""core.loader / core.subject: dynamic import of the user's connector script and its model."""
import sys

import pytest

from _support import BLACK_BOX_CONNECTOR, REPO_ROOT, write_script
from prism.core.loader import load_user_connector
from prism.core.subject import Subject, load_subject
from prism.inspectors.sklearn import SklearnInspector

FULL_CONNECTOR = '''
from core.connectorTemplate import WhydunnitConnector

class MyConnector(WhydunnitConnector):
    def load_model(self):
        return "model"

    def load_item(self, row_id, meta):
        return row_id

    def run_inference(self, model, batch):
        return batch

    def render_item_for_gallery(self, item):
        return {"type": "text", "payload": str(item)}
'''

REQUIRED_ONLY_CONNECTOR = '''
from core.connectorTemplate import WhydunnitConnector

class MinimalConnector(WhydunnitConnector):
    def load_model(self):
        return "model"

    def load_item(self, row_id, meta):
        return row_id

    def run_inference(self, model, batch):
        return batch
'''


def test_loads_connector_and_passes_project_root(tmp_path):
    script = write_script(tmp_path / "connector.py", FULL_CONNECTOR)

    connector = load_user_connector(str(script), str(tmp_path))

    assert type(connector).__name__ == "MyConnector"
    assert connector.project_root == str(tmp_path)
    assert connector.load_model() == "model"


def test_project_root_is_importable_from_the_connector(tmp_path, request):
    module_name = "prism_loader_test_helper"
    request.addfinalizer(lambda: sys.modules.pop(module_name, None))
    project = tmp_path / "project"
    write_script(project / f"{module_name}.py", "ANSWER = 42\n")
    script = write_script(tmp_path / "scripts" / "connector.py", FULL_CONNECTOR.replace(
        'return "model"', f"import {module_name}; return {module_name}.ANSWER"))

    connector = load_user_connector(str(script), str(project))

    assert str(project) in sys.path
    assert connector.load_model() == 42


def test_empty_project_root_is_not_added_to_sys_path(tmp_path):
    script = write_script(tmp_path / "connector.py", FULL_CONNECTOR)
    before = list(sys.path)

    load_user_connector(str(script), "")

    assert sys.path == before


def test_script_that_raises_is_reported_as_import_error(tmp_path):
    script = write_script(tmp_path / "bad.py", "raise RuntimeError('boom at import')\n")

    with pytest.raises(ImportError, match="boom at import"):
        load_user_connector(str(script), "")


def test_syntax_error_is_reported_as_import_error(tmp_path):
    script = write_script(tmp_path / "broken.py", "def oops(:\n")

    with pytest.raises(ImportError):
        load_user_connector(str(script), "")


def test_missing_script_raises_import_error(tmp_path):
    with pytest.raises(ImportError):
        load_user_connector(str(tmp_path / "does_not_exist.py"), "")


def test_script_without_a_connector_class_raises_value_error(tmp_path):
    script = write_script(tmp_path / "plain.py", "class NotAConnector:\n    pass\n")

    with pytest.raises(ValueError, match="No class inheriting"):
        load_user_connector(str(script), "")


def test_connector_written_against_the_snake_case_shim_is_found(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    source = FULL_CONNECTOR.replace("from core.connectorTemplate", "from prism.core.connector_template")
    script = write_script(tmp_path / "connector.py", source)

    connector = load_user_connector(str(script), "")

    assert connector.load_model() == "model"


def test_load_subject_returns_connector_model_and_inspector(tmp_path):
    script = write_script(tmp_path / "connector.py", BLACK_BOX_CONNECTOR)

    subject = load_subject(str(tmp_path), str(script), "Scikit-Learn")

    assert isinstance(subject, Subject)
    assert type(subject.connector).__name__ == "EchoConnector"
    assert subject.model == "black-box-model"
    assert isinstance(subject.inspector, SklearnInspector)
    assert subject.inspector.model == "black-box-model"


def test_load_subject_checks_the_framework_before_running_user_code(tmp_path):
    script = write_script(tmp_path / "connector.py", "raise RuntimeError('should not be imported')\n")

    with pytest.raises(ValueError, match="Framework 'JAX' not found"):
        load_subject(str(tmp_path), str(script), "JAX")


def test_load_subject_accepts_an_explicit_registry(tmp_path):
    script = write_script(tmp_path / "connector.py", BLACK_BOX_CONNECTOR)

    class RecordingInspector:
        def __init__(self, model):
            self.model = model

    subject = load_subject(str(tmp_path), str(script), "Custom", registry={"Custom": RecordingInspector})

    assert isinstance(subject.inspector, RecordingInspector)


def test_render_item_for_gallery_is_optional(tmp_path):
    """The template documents a default for render_item_for_gallery, but marks it abstract."""
    script = write_script(tmp_path / "connector.py", REQUIRED_ONLY_CONNECTOR)

    connector = load_user_connector(str(script), "")

    assert connector.render_item_for_gallery(5) == {"type": "text", "payload": "5"}


def test_most_derived_connector_class_is_chosen(tmp_path):
    source = FULL_CONNECTOR.replace("class MyConnector", "class BaseLab") + '''

class MRIConnector(BaseLab):
    def load_model(self):
        return "mri"
'''
    script = write_script(tmp_path / "connector.py", source)

    connector = load_user_connector(str(script), "")

    assert connector.load_model() == "mri"


def test_both_template_import_paths_yield_the_same_class(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    import core.connectorTemplate as legacy
    import prism.core.connector_template as shim

    assert shim.WhydunnitConnector is legacy.WhydunnitConnector



NEW_STYLE_CONNECTOR = FULL_CONNECTOR.replace("from core.connectorTemplate import WhydunnitConnector",
                                             "from prism import PrismConnector").replace("(WhydunnitConnector)",
                                                                                         "(PrismConnector)")


def test_a_connector_written_against_prism_connector_loads(tmp_path):
    from prism import PrismConnector, WhydunnitConnector
    script = write_script(tmp_path / "connector.py", NEW_STYLE_CONNECTOR)

    connector = load_user_connector(str(script), "")

    assert isinstance(connector, PrismConnector) and WhydunnitConnector is PrismConnector


def test_older_connectors_work_even_when_the_project_has_its_own_core_package(tmp_path, request):
    """The old import path must not take over a `core` package that belongs to the user's project."""
    for name in ("core", "core.settings"):
        request.addfinalizer(lambda name=name: sys.modules.pop(name, None))
    project = tmp_path / "project"
    write_script(project / "core" / "__init__.py", "")
    write_script(project / "core" / "settings.py", "MODEL = 'from the project'\n")
    source = FULL_CONNECTOR.replace('return "model"', "from core.settings import MODEL; return MODEL")
    script = write_script(project / "connector.py", source)

    connector = load_user_connector(str(script), str(project))

    assert connector.load_model() == "from the project"
    from prism import PrismConnector
    assert isinstance(connector, PrismConnector)


def test_a_connector_missing_a_method_is_named(tmp_path):
    source = REQUIRED_ONLY_CONNECTOR.replace("    def run_inference(self, model, batch):\n        return batch\n", "")
    script = write_script(tmp_path / "connector.py", source)

    with pytest.raises(ValueError, match="MinimalConnector does not implement run_inference"):
        load_user_connector(str(script), "")


def test_two_unrelated_connectors_are_refused(tmp_path):
    source = FULL_CONNECTOR + FULL_CONNECTOR.replace("class MyConnector", "class OtherConnector").split(
        "from core.connectorTemplate import WhydunnitConnector")[1]
    script = write_script(tmp_path / "connector.py", source)

    with pytest.raises(ValueError, match="several connectors"):
        load_user_connector(str(script), "")


def test_a_connector_built_on_another_copy_of_the_template_is_explained(tmp_path):
    source = """
from abc import ABC

class WhydunnitConnector(ABC):   # a vendored copy, not PRISM's class
    pass

class MyConnector(WhydunnitConnector):
    pass
"""
    script = write_script(tmp_path / "connector.py", source)

    with pytest.raises(ValueError, match="different copy of the PRISM connector template"):
        load_user_connector(str(script), "")


def test_an_imported_connector_is_used_when_the_script_defines_none(tmp_path, request):
    request.addfinalizer(lambda: sys.modules.pop("lab_connectors", None))
    project = tmp_path / "project"
    write_script(project / "lab_connectors.py", NEW_STYLE_CONNECTOR)
    script = write_script(project / "connector.py", "from lab_connectors import MyConnector\n")

    connector = load_user_connector(str(script), str(project))

    assert type(connector).__name__ == "MyConnector"
