"""
columns.py
----------
What a candidate is told about the columns it receives.

`build_pipeline` gets the columns that are actually present, after the target
and every excluded column have been removed. Building column groups from this
list, instead of from names remembered from the profile, is what lets the
harness drop a column (a leak, an identifier) without the candidate breaking.

    ColumnInfo(name, kind, dtype, n_unique, missing_pct, semantic_type)
    names(columns, *kinds) -> list[str]
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, List, Optional

import pandas as pd

# the profile's roles, which are more useful than dtypes: an integer column can
# be an identifier, a flag, a small ordinal code or a real measurement
KINDS = (
    "numeric",           # continuous measurement
    "discrete_numeric",  # integer with few levels; behaves like a category for many models
    "binary",            # exactly two values, which may be strings
    "categorical",       # string categories
    "datetime",          # dates or times, usually still strings in the frame
    "text",              # free text
    "identifier",        # near-unique; rarely generalises
    "constant",
    "empty",
)
NUMERIC_KINDS = ("numeric", "discrete_numeric")


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    kind: str
    dtype: str = ""
    n_unique: Optional[int] = None
    missing_pct: Optional[float] = None
    semantic_type: Optional[str] = None

    @property
    def is_numeric_dtype(self) -> bool:
        return self.dtype.startswith(("int", "float", "uint", "Int", "Float", "bool"))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ColumnInfo":
        known = {field: data.get(field) for field in cls.__dataclass_fields__}
        known["kind"] = known.get("kind") or "categorical"
        known["dtype"] = known.get("dtype") or ""
        return cls(**known)


def names(columns: Iterable[ColumnInfo], *kinds: str) -> List[str]:
    """Names of the columns whose kind is one of `kinds`, in frame order."""
    wanted = set(kinds)
    return [column.name for column in columns if column.kind in wanted]


def infer_kind(series: pd.Series) -> str:
    """A fallback for a column the profile does not describe."""
    non_null = series.dropna()
    if non_null.empty:
        return "empty"
    unique = non_null.nunique()
    if unique == 1:
        return "constant"
    if unique == 2:
        return "binary"
    if pd.api.types.is_numeric_dtype(series):
        return "discrete_numeric" if pd.api.types.is_integer_dtype(series) and unique <= 15 else "numeric"
    return "categorical"
