"""
features.py
-----------
Tested transformers a candidate may import instead of writing its own.

They are optional: a candidate can build any transformer it likes. These exist
because hand-written date handling was the single largest cause of failed
attempts, and because a class imported from an installed module always pickles
and always supports `get_feature_names_out` and `set_output`.

    DateParts(parts=("year", "month", "day", "dayofweek"))
    FrequencyEncoder()
    CategoryCleaner()
"""

from __future__ import annotations

import warnings
from typing import List, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

DATE_PARTS = (
    "year", "quarter", "month", "week", "day", "dayofweek", "dayofyear", "hour", "minute",
    "is_month_start", "is_month_end", "timestamp",
)
_MISSING = "__missing__"


def _as_frame(X, columns: Sequence) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(np.asarray(X, dtype=object), columns=list(columns))


def _input_names(X) -> List:
    if isinstance(X, pd.DataFrame):
        return list(X.columns)
    return [f"x{position}" for position in range(np.asarray(X).shape[1])]


def parse_dates(values: pd.Series) -> pd.Series:
    """Datetime values from strings, whatever their format; unparseable -> NaT."""
    # utc=True makes mixed offsets parse to one dtype; dropping the zone after
    # keeps naive values at their original wall-clock time
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if pd.api.types.is_datetime64_any_dtype(values):
            parsed = pd.to_datetime(values, utc=True)
        else:
            parsed = pd.to_datetime(values, errors="coerce", format="ISO8601", utc=True)
            if parsed.isna().sum() > values.isna().sum():
                # mixed or non-ISO formats, including time-only strings such as 11:30:55
                parsed = pd.to_datetime(values, errors="coerce", format="mixed", utc=True)
    return parsed.dt.tz_convert(None)


class DateParts(BaseEstimator, TransformerMixin):
    """Expand date or time columns into numeric parts.

    Each input column becomes one output column per part, named
    `<column>__<part>`. Values that do not parse become NaN, so follow this
    with an imputer for models that cannot take missing values.
    """

    def __init__(self, parts: Sequence[str] = ("year", "month", "day", "dayofweek")):
        self.parts = parts

    def fit(self, X, y=None):
        unknown = [part for part in self.parts if part not in DATE_PARTS]
        if unknown:
            raise ValueError(f"unknown date parts {unknown}; choose from {list(DATE_PARTS)}")
        self.feature_names_in_ = np.array(_input_names(X), dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X):
        frame = _as_frame(X, self.feature_names_in_)
        output = {}
        for column in self.feature_names_in_:
            dates = parse_dates(frame[column])
            for part in self.parts:
                name = f"{column}__{part}"
                if part == "timestamp":
                    values = (dates - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)
                    output[name] = values.astype(float)
                elif part == "week":
                    # ISO week; unparseable dates stay missing rather than raising
                    weeks = dates.dt.isocalendar().week
                    output[name] = pd.Series(weeks.to_numpy(dtype=float, na_value=np.nan), index=frame.index)
                else:
                    output[name] = getattr(dates.dt, part).astype(float)
        return pd.DataFrame(output, index=frame.index)

    def get_feature_names_out(self, input_features=None):
        columns = self.feature_names_in_ if input_features is None else input_features
        return np.array([f"{column}__{part}" for column in columns for part in self.parts], dtype=object)


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replace each category by its share of the training rows.

    Suited to high-cardinality columns where one-hot encoding explodes. A
    category unseen during fit, including missing values when none were seen,
    becomes 0.
    """

    def __init__(self, normalize: bool = True):
        self.normalize = normalize

    def fit(self, X, y=None):
        self.feature_names_in_ = np.array(_input_names(X), dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        frame = _as_frame(X, self.feature_names_in_)
        self.frequencies_ = {}
        for column in self.feature_names_in_:
            values = frame[column].astype(object).where(frame[column].notna(), _MISSING)
            self.frequencies_[column] = values.value_counts(normalize=self.normalize).to_dict()
        return self

    def transform(self, X):
        frame = _as_frame(X, self.feature_names_in_)
        output = {}
        for column in self.feature_names_in_:
            values = frame[column].astype(object).where(frame[column].notna(), _MISSING)
            output[f"{column}__freq"] = values.map(self.frequencies_[column]).fillna(0.0).astype(float)
        return pd.DataFrame(output, index=frame.index)

    def get_feature_names_out(self, input_features=None):
        columns = self.feature_names_in_ if input_features is None else input_features
        return np.array([f"{column}__freq" for column in columns], dtype=object)


class CategoryCleaner(BaseEstimator, TransformerMixin):
    """Make labels that differ only by case or spacing one category.

    Strips surrounding spaces, collapses inner runs of whitespace and lowercases,
    so 'Basic', ' Basic' and 'BASIC' encode as one category. Missing values stay
    missing. Place it before the encoder of the categorical columns it cleans.
    """

    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase

    def fit(self, X, y=None):
        self.feature_names_in_ = np.array(_input_names(X), dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X):
        frame = _as_frame(X, self.feature_names_in_)
        output = {}
        for column in self.feature_names_in_:
            values = frame[column]
            cleaned = values.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
            if self.lowercase:
                cleaned = cleaned.str.lower()
            output[column] = cleaned.astype(object).where(values.notna(), np.nan)
        return pd.DataFrame(output, index=frame.index)

    def get_feature_names_out(self, input_features=None):
        columns = self.feature_names_in_ if input_features is None else input_features
        return np.array(list(columns), dtype=object)
