"""PRISM: audits medical-imaging (and other) classifiers for shortcut learning and demographic bias.

Write a connector for your model by subclassing PrismConnector (see prism/connector.py), then run
`prism` for the GUI, `prism audit <results dir>` for a report, or `prism run --config <job>` on a
compute node.
"""
from prism.connector import PrismConnector, WhydunnitConnector  # noqa: F401

__version__ = "1.0.0"
__all__ = ["PrismConnector", "WhydunnitConnector", "__version__"]
