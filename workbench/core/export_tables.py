"""Combine compact export tables without changing analysis results."""
from __future__ import annotations

import pandas as pd


def summary_table(sections: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Keep scalar results together, with record IDs for paired measurements.

    Key/value tables retain extra metadata such as units and fit uncertainty.
    Wider tables become one parameter/value row per field, in source row order.
    """
    columns = ["Section", "Record", "Parameter", "Value"]
    rows = []
    for section, table in sections.items():
        key = next((name for name in ("Parameter", "Quantity", "Label")
                    if name in table and "Value" in table), None)
        if key is not None:
            metadata = [name for name in table.columns if name not in (key, "Value")]
            columns.extend(name for name in metadata if name not in columns)
            for values in table.itertuples(index=False, name=None):
                source = dict(zip(table.columns, values))
                rows.append({"Section": section, "Record": "",
                             "Parameter": source[key], "Value": source["Value"],
                             **{name: source[name] for name in metadata}})
        else:
            for record, values in enumerate(table.itertuples(index=False, name=None), start=1):
                rows.extend({"Section": section, "Record": record if len(table) > 1 else "",
                             "Parameter": parameter, "Value": value}
                            for parameter, value in zip(table.columns, values))
    # Inference could coerce mixed integer/float values and lose large integers.
    return pd.DataFrame(rows, columns=columns, dtype=object)
