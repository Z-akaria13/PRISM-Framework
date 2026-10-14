"""
SUBJECT LOADING
The model under audit: the user's connector, the model it loads, and the
framework inspector wrapped around it. Shared by the Setup tab and headless.py.
"""
from dataclasses import dataclass
from typing import Any

from prism.inspectors import get_inspectors

from .loader import load_user_connector


@dataclass
class Subject:
    connector: Any
    model: Any
    inspector: Any


def load_subject(project_root, script_path, framework, registry=None) -> Subject:
    registry = get_inspectors() if registry is None else registry
    if framework not in registry:
        raise ValueError(f"Framework '{framework}' not found in Inspectors folder.")

    connector = load_user_connector(script_path, project_root)
    model = connector.load_model()
    return Subject(connector=connector, model=model, inspector=registry[framework](model))
