"""Older snake-case import path for the connector; the same class as prism.connector.PrismConnector."""
from prism.connector import PrismConnector, WhydunnitConnector  # noqa: F401

__all__ = ["PrismConnector", "WhydunnitConnector"]
