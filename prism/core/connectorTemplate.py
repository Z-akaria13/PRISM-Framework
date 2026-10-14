"""Older import path for the connector, kept so existing connectors keep working.

    from core.connectorTemplate import WhydunnitConnector       # older connectors
    from prism import PrismConnector                             # new connectors

Both names are the same class, defined in prism/connector.py.
"""
from prism.connector import PrismConnector, WhydunnitConnector  # noqa: F401

__all__ = ["PrismConnector", "WhydunnitConnector"]
