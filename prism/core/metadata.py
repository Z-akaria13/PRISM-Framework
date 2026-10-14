"""
METADATA CSV
How PRISM reads a metadata table. The ID column is read as text: pandas would otherwise turn an
ID such as 0042 into the number 42, and the connector would be asked for item "42".
"""
import pandas as pd


def read_metadata(path, id_col=None):
    """The CSV at `path`, with `id_col` (if given and present) as text and every other column
    typed as pandas infers it."""
    return pd.read_csv(path, dtype={id_col: str} if id_col else None)
