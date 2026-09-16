"""
summary.py
----------
Produce a structured description of a tabular dataset that is rich enough for a
downstream agent (or human) to plan preprocessing and modelling WITHOUT seeing
the rows themselves.

Two entry points:
    profile_dataset(df, target=...) -> dict          # machine readable
    render_profile(profile)         -> str           # compact text for an LLM

Only depends on pandas + numpy.
"""

from __future__ import annotations

import json
import math
import re
import warnings
from typing import Any, Callable

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# tuning knobs
# ----------------------------------------------------------------------------

ID_UNIQUENESS = 0.98          # unique_ratio above this -> looks like an identifier
HIGH_CARDINALITY = 50         # distinct categories above this -> needs special encoding
RARE_CATEGORY_PCT = 1.0       # categories below this share are "rare"
LEAKAGE_ASSOC = 0.95          # association with target above this -> suspicious
REDUNDANT_CORR = 0.95         # |spearman| above this -> near-duplicate feature pair
SKEW_FLAG = 1.0               # |skew| above this -> consider a transform
EXTREME_RANGE = 6             # orders of magnitude a skewed column spans before it is a medium flag
IMBALANCE_FLAG = 10.0         # majority/minority ratio above this -> imbalanced
TEXT_MEAN_LEN = 40            # mean char length above this -> free text, not category
DATE_PARSE_SUCCESS = 0.85     # object columns above this parse rate -> datetime-like
DISCRETE_LEVELS = 15          # integer columns with this many values or fewer behave like categories
SEMANTIC_SAMPLE = 1000        # cap expensive string semantic checks
COORDINATE_VALID = 99.5       # coordinate columns below this % in range are flagged
MAX_CONTINGENCY = 5_000_000   # cells; a cross-tab bigger than this is skipped, not built

DATE_NAME_TOKENS = {
    "date", "time", "datetime", "timestamp", "created", "updated", "year", "month", "day", "dob",
}


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------

def _r(x: Any, nd: int = 4) -> Any:
    """Round floats, pass everything else through, turn nan/inf into None."""
    if x is None:
        return None
    if isinstance(x, (np.floating, float)):
        if math.isnan(x) or math.isinf(x):
            return None
        return round(float(x), nd)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    return x


def _safe_str(x: Any, limit: int = 40) -> str:
    s = str(x)
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _name_tokens(name: Any) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", str(name).lower()) if t}


def _sample_non_null(s: pd.Series, n: int = SEMANTIC_SAMPLE) -> pd.Series:
    non_null = s.dropna()
    if len(non_null) <= n:
        return non_null
    return non_null.sample(n, random_state=0)


def _to_datetime_silent(s: pd.Series) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return pd.to_datetime(s, errors="coerce", format="mixed")


def _datetime_parse_rate(s: pd.Series) -> float:
    sample = _sample_non_null(s)
    if sample.empty or pd.api.types.is_numeric_dtype(sample):
        return 0.0
    values = sample.astype(str).str.strip()
    values = values[values != ""]
    if values.empty:
        return 0.0
    date_name = bool(_name_tokens(s.name) & DATE_NAME_TOKENS)
    dateish_values = values.str.contains(
        r"(?:\d{4}[-/]\d{1,2})|(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4})|(?:\d{1,2}:\d{2})",
        regex=True,
    ).mean()
    if not date_name and dateish_values < 0.5:
        return 0.0
    return float(_to_datetime_silent(values).notna().mean())


def _coordinate_valid_pct(s: pd.Series, kind: str) -> float | None:
    v = pd.to_numeric(s.dropna(), errors="coerce").dropna()
    if v.empty:
        return None
    lo, hi = (-90, 90) if kind == "latitude" else (-180, 180)
    return float(100 * v.between(lo, hi).mean())


def _magnitude(lo: float, hi: float) -> float | None:
    """Orders of magnitude a range spans; how many a linear model has to absorb."""
    spread = hi - lo
    return _r(math.log10(spread), 1) if spread > 0 else None


RateFn = Callable[[pd.Series], float]


def _date_rate_cache() -> RateFn:
    """Parse-rate per column, computed once. Role and semantic inference both ask."""
    cache: dict[Any, float] = {}

    def rate(s: pd.Series) -> float:
        if s.name not in cache:
            cache[s.name] = _datetime_parse_rate(s)
        return cache[s.name]

    return rate


def _infer_semantic_type(s: pd.Series, role: str, date_rate: RateFn) -> str | None:
    """
    Add a domain hint without changing the modelling role. This keeps the
    profiler general while making recommendations more specific.
    """
    tokens = _name_tokens(s.name)
    name = "_".join(tokens)

    if role in ("empty", "constant"):
        return None

    if {"lat", "latitude"} & tokens:
        return "latitude"
    if {"lon", "long", "lng", "longitude"} & tokens:
        return "longitude"
    if {"address", "addr", "street"} & tokens:
        return "address"
    if {"city", "state", "country", "county", "province", "zip", "zipcode", "postal"} & tokens:
        return "location"
    if {"email", "e_mail"} & tokens:
        return "email"
    if {"url", "uri", "link", "website", "web"} & tokens:
        return "url"

    if role == "datetime":
        return "datetime"
    if role == "text":
        return "free_text"
    if pd.api.types.is_numeric_dtype(s):
        return None

    values = _sample_non_null(s).astype(str).str.strip()
    if values.empty:
        return None

    lower = values.str.lower()
    if lower.str.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$").mean() >= 0.8:
        return "email"
    if lower.str.match(r"^(https?://|www\.)").mean() >= 0.8:
        return "url"
    if values.str.match(r"^\s*[\{\[]").mean() >= 0.8:
        return "json_like"
    if date_rate(s) >= DATE_PARSE_SUCCESS:
        return "datetime"
    if values.str.match(r"^[a-f0-9]{16,}$", case=False).mean() >= 0.8:
        return "hash_like"
    if values.str.match(r"^[a-z0-9][a-z0-9_-]{5,}$", case=False).mean() >= 0.8 and role == "identifier":
        return "code_like"
    if "id" in tokens or name.endswith("_id") or name.startswith("id_"):
        return "id"

    return None


# ----------------------------------------------------------------------------
# semantic role inference
# ----------------------------------------------------------------------------

def _infer_role(s: pd.Series, date_rate: RateFn) -> str:
    """
    Classify a column into a modelling-relevant role. This is deliberately more
    useful than the raw dtype: an int column can be an id, a boolean, an
    ordinal code or a genuine measurement.
    """
    non_null = s.dropna()
    n = len(non_null)
    if n == 0:
        return "empty"

    nunique = non_null.nunique()
    if nunique == 1:
        return "constant"

    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"

    numeric = pd.api.types.is_numeric_dtype(s)
    if not numeric and date_rate(s) >= DATE_PARSE_SUCCESS:
        return "datetime"

    # long strings are text even when there are only a couple of distinct
    # values. Sampled: the mean length of a million strings is the same as
    # the mean length of a thousand of them.
    is_long_text = (
        not numeric
        and _sample_non_null(non_null).astype(str).str.len().mean() > TEXT_MEAN_LEN
    )

    if not is_long_text and (pd.api.types.is_bool_dtype(s) or nunique == 2):
        return "binary"

    if numeric:
        if pd.api.types.is_integer_dtype(s):
            if nunique / n >= ID_UNIQUENESS:
                return "identifier"
            # small integer domain behaves like a category for most models
            if nunique <= DISCRETE_LEVELS:
                return "discrete_numeric"
        return "numeric"

    # object / string
    if is_long_text:
        return "text"
    if nunique / n >= ID_UNIQUENESS:
        return "identifier"
    return "categorical"


def _target_role(s: pd.Series, role: str, declared: str | None) -> str:
    """The feature roles, re-read for a column that is the label.

    A near-unique integer column is an identifier when it is a feature and a
    regression target when it is the label: prices in whole rupees, salaries,
    counts. Left as "identifier" it was profiled as a classification target
    with thousands of classes, and every stage downstream planned for that.

    `declared` is what the caller says the target is ("numeric" or
    "categorical"), which settles the one genuinely ambiguous case: a handful
    of integer levels, which could be a rating to regress or a code to classify.
    """
    numeric = pd.api.types.is_numeric_dtype(s)
    if declared == "categorical":
        return "discrete_numeric" if numeric and role in ("numeric", "identifier") else role
    if role == "identifier" and numeric:
        return "numeric"
    if declared == "numeric" and role == "discrete_numeric":
        return "numeric"
    return role


# ----------------------------------------------------------------------------
# association measures (feature <-> target)
# ----------------------------------------------------------------------------

def _contingency(a: pd.Series, b: pd.Series) -> np.ndarray | None:
    """Cross-tab as a dense array, built from integer codes.

    pd.crosstab on a column with tens of thousands of levels runs a
    pure-Python group loop and took most of the profile's wall time; two
    factorize calls and a bincount are the same table in a fraction of it.
    """
    a_codes, _ = pd.factorize(a)
    b_codes, _ = pd.factorize(b)
    keep = (a_codes >= 0) & (b_codes >= 0)
    a_codes, b_codes = a_codes[keep], b_codes[keep]
    if a_codes.size == 0:
        return None
    r, k = int(a_codes.max()) + 1, int(b_codes.max()) + 1
    if r * k > MAX_CONTINGENCY:
        return None
    tab = np.bincount(a_codes * k + b_codes, minlength=r * k).reshape(r, k).astype(float)
    # levels that only ever sat next to a null leave empty rows; they are not levels
    return tab[tab.sum(1) > 0][:, tab.sum(0) > 0]


def _cramers_v(a: pd.Series, b: pd.Series) -> float | None:
    """Categorical vs categorical, bias-corrected. Returns 0..1."""
    tab = _contingency(a, b)
    if tab is None or tab.shape[0] < 2 or tab.shape[1] < 2:
        return None
    n = tab.sum()
    if n < 2:
        return None
    expected = np.outer(tab.sum(1), tab.sum(0)) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum(np.where(expected > 0, (tab - expected) ** 2 / expected, 0.0))
    phi2 = chi2 / n
    r, k = tab.shape
    phi2corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    if denom <= 0:
        return None
    return float(np.sqrt(phi2corr / denom))


def _correlation_ratio(categories: pd.Series, values: pd.Series) -> float | None:
    """
    Eta: numeric vs categorical. How much of the numeric variance is explained
    by group membership. Returns 0..1.

    Bias-corrected (epsilon squared). Raw eta grows with the number of groups
    whether or not they mean anything: a column with one row per level
    explains the target perfectly and scored 1.0, which read as leakage.
    """
    codes, _ = pd.factorize(categories)
    v = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    keep = (codes >= 0) & np.isfinite(v)
    codes, v = codes[keep], v[keep]
    n = v.size
    if n < 3:
        return None

    counts = np.bincount(codes)
    k = int((counts > 0).sum())
    # one row per group explains everything and says nothing
    if k < 2 or n <= k:
        return None

    grand = v.mean()
    total = ((v - grand) ** 2).sum()
    if total == 0:
        return None
    sums = np.bincount(codes, weights=v, minlength=counts.size)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    between = (counts * (means - grand) ** 2).sum()
    eta2 = between / total
    adjusted = 1 - (1 - eta2) * (n - 1) / (n - k)
    return float(np.sqrt(max(0.0, adjusted)))


def _association(feature: pd.Series, target: pd.Series,
                 f_role: str, t_role: str) -> tuple[str, float | None]:
    """
    Return (method, strength 0..1). Strength is comparable across types, which
    is what makes it usable as a leakage screen.
    """
    f_num = f_role in ("numeric", "discrete_numeric")
    t_num = t_role in ("numeric", "discrete_numeric")

    try:
        if f_num and t_num:
            c = feature.corr(target, method="spearman")
            return "spearman", None if pd.isna(c) else abs(float(c))
        if f_num and not t_num:
            return "correlation_ratio", _correlation_ratio(target, feature)
        if (not f_num) and t_num:
            return "correlation_ratio", _correlation_ratio(feature, target)
        return "cramers_v", _cramers_v(feature, target)
    except Exception:
        return "none", None


# ----------------------------------------------------------------------------
# per-column profiling
# ----------------------------------------------------------------------------

def _category_summary(non_null: pd.Series, include_values: bool, max_categories: int) -> dict:
    counts = non_null.value_counts()
    top = counts.head(max_categories)
    out: dict[str, Any] = {
        "top_categories": [
            {
                "value": _safe_str(k) if include_values else f"<cat_{i}>",
                "pct": _r(100 * v / len(non_null), 2),
            }
            for i, (k, v) in enumerate(top.items())
        ],
        "rare_categories": int((100 * counts / len(non_null) < RARE_CATEGORY_PCT).sum()),
    }
    if len(counts) > 1:
        out["imbalance_ratio"] = _r(counts.iloc[0] / counts.iloc[-1], 2)
    return out


def _label_variants(non_null: pd.Series, examples: int = 3) -> dict | None:
    """Labels that are one category once case and spacing are ignored.

    'Basic' and ' Basic', or 'LOS ANGELES' and 'Los Angeles', are one category
    written two ways; an encoder treats them as two, which splits the rows and
    their signal between columns that mean the same thing.
    """
    labels = pd.Series(non_null.astype(str).unique())
    normalized = labels.str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    groups = [sorted(group) for group in labels.groupby(normalized).agg(list) if len(group) > 1]
    if not groups:
        return None
    return {"groups": len(groups), "examples": groups[:examples]}


def _numeric_stats(v: pd.Series) -> dict:
    q1, q3 = v.quantile(0.25), v.quantile(0.75)
    iqr = q3 - q1
    outliers = ((v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)).sum() if iqr > 0 else 0
    return {
        "mean": _r(v.mean()), "std": _r(v.std()),
        "min": _r(v.min()), "p25": _r(q1), "median": _r(v.median()),
        "p75": _r(q3), "max": _r(v.max()),
        "skew": _r(v.skew()), "kurtosis": _r(v.kurt()),
        "zeros_pct": _r(100 * (v == 0).mean(), 2),
        "negatives_pct": _r(100 * (v < 0).mean(), 2),
        "iqr_outlier_pct": _r(100 * outliers / len(v), 2),
    }


def _profile_column(s: pd.Series, role: str, semantic_type: str | None, n_rows: int,
                    include_values: bool, max_categories: int) -> dict:
    non_null = s.dropna()
    n_missing = int(s.isna().sum())
    nunique = int(non_null.nunique())

    col: dict[str, Any] = {
        "dtype": str(s.dtype),
        "role": role,
        "missing_pct": _r(100 * n_missing / n_rows if n_rows else 0, 2),
        "n_unique": nunique,
        "unique_ratio": _r(nunique / len(non_null), 4) if len(non_null) else None,
    }
    if semantic_type:
        col["semantic_type"] = semantic_type

    if role in ("numeric", "discrete_numeric", "binary") and pd.api.types.is_numeric_dtype(s):
        if len(non_null):
            v = non_null.astype(float)
            col["stats"] = _numeric_stats(v)
            # order of magnitude matters for whether scaling is required
            col["magnitude"] = _magnitude(v.min(), v.max())
            if semantic_type in ("latitude", "longitude"):
                col["coordinate_valid_pct"] = _r(_coordinate_valid_pct(s, semantic_type), 2)
            if role == "binary":
                col.update(_category_summary(non_null, include_values, max_categories))

    elif role == "identifier":
        # deliberately do not enumerate: it is noise, it is large, and it is
        # the most personally identifying part of most tables
        col["note"] = "values not listed (identifier-like)"

    elif role in ("categorical", "binary"):
        col.update(_category_summary(non_null, include_values, max_categories))

    if role in ("categorical", "binary") and not pd.api.types.is_numeric_dtype(s) and len(non_null):
        variants = _label_variants(non_null)
        if variants:
            col["label_variants"] = variants

    elif role == "datetime":
        dates = non_null if pd.api.types.is_datetime64_any_dtype(s) else _to_datetime_silent(non_null).dropna()
        col["stats"] = {
            "min": str(dates.min()) if include_values and len(dates) else None,
            "max": str(dates.max()) if include_values and len(dates) else None,
            "is_monotonic": bool(dates.is_monotonic_increasing) if len(dates) else None,
            "span_days": _r((dates.max() - dates.min()).days) if len(dates) > 1 else None,
            "parse_success_pct": _r(100 * len(dates) / len(non_null), 2) if len(non_null) else None,
        }
        if not include_values:
            col["stats"]["min"] = "<redacted>"
            col["stats"]["max"] = "<redacted>"

    elif role == "text":
        lengths = non_null.astype(str).str.len()
        words = non_null.astype(str).str.split().str.len()
        col["stats"] = {
            "mean_chars": _r(lengths.mean(), 1),
            "max_chars": _r(lengths.max()),
            "mean_words": _r(words.mean(), 1),
        }

    return col


# ----------------------------------------------------------------------------
# target profiling
# ----------------------------------------------------------------------------

def _profile_target(s: pd.Series, role: str, include_values: bool) -> dict:
    non_null = s.dropna()
    out: dict[str, Any] = {
        "name": s.name,
        "dtype": str(s.dtype),
        "role": role,
        "missing_pct": _r(100 * s.isna().mean(), 2),
        "n_unique": int(non_null.nunique()),
    }

    if role == "numeric":
        task = "regression"
        if len(non_null):
            v = non_null.astype(float)
            out["stats"] = _numeric_stats(v)
            out["magnitude"] = _magnitude(v.min(), v.max())
    else:
        counts = non_null.value_counts()
        task = "binary_classification" if len(counts) == 2 else "multiclass_classification"
        out["n_classes"] = int(len(counts))
        out["class_distribution"] = [
            {
                "class": _safe_str(k) if include_values else f"<class_{i}>",
                "count": int(v),
                "pct": _r(100 * v / len(non_null), 2),
            }
            for i, (k, v) in enumerate(counts.head(30).items())
        ]
        if len(counts) > 1:
            out["imbalance_ratio"] = _r(counts.iloc[0] / counts.iloc[-1], 2)
            out["minority_count"] = int(counts.iloc[-1])
        # the score any model must beat to be worth anything
        out["majority_class_baseline"] = _r(100 * counts.iloc[0] / len(non_null), 2) if len(counts) else None

    out["task_type"] = task
    return out


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def profile_dataset(
    df: pd.DataFrame,
    target: str | None = None,
    schema: dict[str, str] | None = None,
    include_values: bool = True,
    max_categories: int = 10,
    corr_sample: int = 20000,
    target_type: str | None = None,
) -> dict:
    """
    Parameters
    ----------
    df              : the dataframe (ideally the TRAINING split only)
    target          : name of the target column, if there is one
    schema          : optional {column: human description}, passed through
    include_values  : False redacts actual category labels / dates
    max_categories  : how many category levels to list per column
    corr_sample     : row cap for the pairwise correlation scan
    target_type     : "numeric" or "categorical" when the caller knows which the
                      target is; settles regression vs classification for an
                      integer target with few levels
    """
    n_rows, n_cols = df.shape
    date_rate = _date_rate_cache()
    roles = {c: _infer_role(df[c], date_rate) for c in df.columns}
    if target is not None and target in df.columns:
        roles[target] = _target_role(df[target], roles[target], target_type)
    semantic_types = {c: _infer_semantic_type(df[c], roles[c], date_rate) for c in df.columns}

    duplicated = df.duplicated()
    profile: dict[str, Any] = {
        "dataset": {
            "n_rows": n_rows,
            "n_columns": n_cols,
            "memory_mb": _r(df.memory_usage(deep=True).sum() / 1e6, 2),
            "duplicate_rows": int(duplicated.sum()),
            "duplicate_rows_pct": _r(100 * duplicated.mean(), 2),
            "complete_rows_pct": _r(100 * df.notna().all(axis=1).mean(), 2),
            "total_missing_pct": _r(100 * df.isna().to_numpy().mean(), 2),
            "values_redacted": not include_values,
        },
        "target": None,
        "columns": {},
        "relationships": {},
        "flags": [],
        "modelling_notes": {},
    }
    if target is not None and target not in df.columns:
        profile["flags"].append({"severity": "high", "column": target,
                                 "issue": "target column not found",
                                 "action": "choose an existing target column or run as unsupervised profiling"})

    # --- target -------------------------------------------------------------
    if target is not None and target in df.columns:
        profile["target"] = _profile_target(df[target], roles[target], include_values)
        if semantic_types[target]:
            profile["target"]["semantic_type"] = semantic_types[target]
        if target_type:
            profile["target"]["declared_type"] = target_type
        if schema and target in schema:
            profile["target"]["description"] = schema[target]

    # --- columns ------------------------------------------------------------
    feature_cols = [c for c in df.columns if c != target]
    for c in feature_cols:
        col = _profile_column(df[c], roles[c], semantic_types[c], n_rows, include_values, max_categories)
        if schema and c in schema:
            col["description"] = schema[c]
        profile["columns"][c] = col

    # --- association with target -------------------------------------------
    # identifiers are left out: a column with a level per row explains any
    # target perfectly and would top the list as a leak every time
    if profile["target"] is not None:
        t_role = roles[target]
        assoc = {}
        for c in feature_cols:
            if roles[c] in ("empty", "constant", "text", "identifier"):
                continue
            method, strength = _association(df[c], df[target], roles[c], t_role)
            if strength is not None:
                assoc[c] = {"method": method, "strength": _r(strength, 3)}
        profile["relationships"]["target_association"] = dict(
            sorted(assoc.items(), key=lambda kv: kv[1]["strength"], reverse=True)
        )

    # --- missingness as a signal -------------------------------------------
    # A column that is only populated for one class leaks the target through
    # its null pattern alone, and no value-based association will ever see it.
    if profile["target"] is not None:
        t_role = roles[target]
        miss_assoc = {}
        for c in feature_cols:
            n_miss = df[c].isna().sum()
            if n_miss == 0 or n_miss == n_rows:
                continue
            indicator = df[c].isna().astype(int)
            method, strength = _association(indicator, df[target], "binary", t_role)
            if strength is not None and strength > 0.1:
                miss_assoc[c] = {"method": method, "strength": _r(strength, 3)}
        if miss_assoc:
            profile["relationships"]["missingness_association"] = dict(
                sorted(miss_assoc.items(), key=lambda kv: kv[1]["strength"], reverse=True)
            )

    # --- redundant feature pairs -------------------------------------------
    num_cols = [c for c in feature_cols if roles[c] in ("numeric", "discrete_numeric")]
    if len(num_cols) > 1:
        sample = df[num_cols].sample(min(len(df), corr_sample), random_state=0)
        corr = sample.corr(method="spearman", numeric_only=True).abs()
        pairs = []
        for i, a in enumerate(corr.columns):
            for b in corr.columns[i + 1:]:
                v = corr.loc[a, b]
                if pd.notna(v) and v >= REDUNDANT_CORR:
                    pairs.append({"a": a, "b": b, "spearman_abs": _r(v, 3)})
        profile["relationships"]["redundant_pairs"] = sorted(
            pairs, key=lambda p: p["spearman_abs"], reverse=True
        )[:20]

    # --- flags --------------------------------------------------------------
    profile["flags"].extend(_build_flags(df, profile, roles, semantic_types, target))
    profile["flags"] = _sort_flags(profile["flags"])

    # --- modelling notes ----------------------------------------------------
    profile["modelling_notes"] = _build_notes(profile, roles, semantic_types, target)

    return profile


def _sort_flags(flags: list[dict]) -> list[dict]:
    order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    return sorted(flags, key=lambda f: order[f["severity"]])


def _coordinate_swap(df: pd.DataFrame, semantic_types: dict) -> dict | None:
    """One latitude and one longitude column whose values fit better exchanged.

    Every generated script for a dataset with this problem wrote its own swap
    heuristic, and several of them crashed doing it; saying it once here is
    cheaper than having five models rediscover it.
    """
    lats = [c for c, t in semantic_types.items() if t == "latitude"]
    lons = [c for c, t in semantic_types.items() if t == "longitude"]
    if len(lats) != 1 or len(lons) != 1:
        return None
    lat, lon = lats[0], lons[0]

    as_is = [_coordinate_valid_pct(df[lat], "latitude"), _coordinate_valid_pct(df[lon], "longitude")]
    swapped = [_coordinate_valid_pct(df[lon], "latitude"), _coordinate_valid_pct(df[lat], "longitude")]
    if any(v is None for v in as_is + swapped):
        return None
    labelled, exchanged = min(as_is), min(swapped)
    if labelled >= COORDINATE_VALID or exchanged <= labelled:
        return None
    return {
        "severity": "medium", "column": lat,
        "issue": (f"{lat}/{lon} may be swapped: {_r(100 - labelled, 2)}% of rows fall outside the "
                  f"valid range as labelled, {_r(100 - exchanged, 2)}% with the two columns exchanged"),
        "action": ("treat the pair as (lon, lat) or drop the out-of-range rows' coordinates before "
                   "deriving distance or region features; tree models can use the raw values either way"),
    }


def _target_flags(tgt: dict) -> list[dict]:
    """A label that cannot be learned from, or one that needs a transform."""
    flags = []
    name, role = tgt["name"], tgt["role"]

    if role == "constant":
        flags.append({"severity": "critical", "column": name,
                      "issue": "target has a single value", "action": "nothing to learn; choose a different target"})
    elif role == "empty":
        flags.append({"severity": "critical", "column": name,
                      "issue": "target is entirely missing", "action": "choose a different target"})
    elif role == "identifier":
        flags.append({"severity": "critical", "column": name,
                      "issue": "target is unique per row and not numeric; it looks like an identifier, not a label",
                      "action": "choose a different target"})
    elif role == "text":
        flags.append({"severity": "critical", "column": name,
                      "issue": "target is free text", "action": "derive a class label or a number from it first"})
    elif role == "datetime":
        flags.append({"severity": "high", "column": name,
                      "issue": "target is a timestamp",
                      "action": "predicting a time is not supported; derive a duration or a class label"})
    elif role == "discrete_numeric" and not tgt.get("declared_type"):
        flags.append({"severity": "low", "column": name,
                      "issue": f"target has {tgt['n_unique']} distinct integer values and is treated as classification",
                      "action": "declare the target numeric if these are quantities to regress rather than codes to classify"})

    if tgt["task_type"] == "regression":
        st = tgt.get("stats") or {}
        skew = abs(st.get("skew") or 0)
        if skew > SKEW_FLAG:
            if (st.get("negatives_pct") or 0) > 0:
                transform = ("a quantile or yeo-johnson target transform (the target has negative values, "
                             "so log1p is undefined)")
            else:
                transform = "TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1)"
            flags.append({
                "severity": "medium", "column": name,
                "issue": f"target skew {st['skew']} across {tgt.get('magnitude')} orders of magnitude",
                "action": (f"consider {transform} for linear and distance-based models so the tail does not "
                           "dominate the fit; tree models can fit the raw target. A log target optimises "
                           "relative error, so confirm the primary metric still improves in original units"),
            })

    if (tgt.get("imbalance_ratio") or 0) > IMBALANCE_FLAG:
        flags.append({"severity": "high", "column": name,
                      "issue": f"class imbalance {tgt['imbalance_ratio']}:1",
                      "action": "stratify the split, use PR-AUC/F1 not accuracy, consider class weights"})
    if (tgt.get("missing_pct") or 0) > 0:
        flags.append({"severity": "high", "column": name,
                      "issue": f"target missing in {tgt['missing_pct']}% of rows",
                      "action": "drop those rows, never impute the target"})
    return flags


def _build_flags(df, profile, roles, semantic_types, target) -> list[dict]:
    """Things a downstream agent should act on, most severe first."""
    flags = []

    for c, role in roles.items():
        if c == target:
            continue
        if role == "constant":
            flags.append({"severity": "high", "column": c,
                          "issue": "constant", "action": "drop"})
        elif role == "empty":
            flags.append({"severity": "high", "column": c,
                          "issue": "entirely missing", "action": "drop"})
        elif role == "identifier":
            stype = semantic_types.get(c)
            action = (
                "drop; direct identifiers rarely generalise and can leak row identity"
                if stype in ("id", "hash_like", "code_like") else
                "drop unless deliberately engineered as grouped history/count features"
            )
            flags.append({"severity": "high", "column": c,
                          "issue": "near-unique, looks like an identifier",
                          "action": action})

    for c, column in profile["columns"].items():
        variants = column.get("label_variants")
        if variants:
            example = " / ".join(repr(label) for label in variants["examples"][0])
            flags.append({"severity": "medium", "column": c,
                          "issue": f"{variants['groups']} categories are written more than one way, "
                                   f"differing only by case or spacing (e.g. {example})",
                          "action": "normalise labels before encoding, e.g. automl_runtime.CategoryCleaner"})

    swap = _coordinate_swap(df, semantic_types)
    if swap:
        flags.append(swap)

    for c, col in profile["columns"].items():
        if roles.get(c) == "empty":
            continue  # already flagged above, do not double report
        stype = semantic_types.get(c)
        if (col.get("missing_pct") or 0) > 50:
            flags.append({"severity": "high", "column": c,
                          "issue": f"{col['missing_pct']}% missing",
                          "action": "drop or use a missingness indicator"})
        elif (col.get("missing_pct") or 0) > 5:
            flags.append({"severity": "medium", "column": c,
                          "issue": f"{col['missing_pct']}% missing",
                          "action": "impute inside the fitted pipeline, not before the split"})
        if col["role"] == "categorical" and col["n_unique"] > HIGH_CARDINALITY:
            if stype in ("address", "location"):
                action = ("extract stable location components or use frequency/target encoding; "
                          "avoid raw one-hot on full addresses")
            elif stype in ("email", "url"):
                action = "extract domain or structural features; do not model raw values directly"
            elif stype == "json_like":
                action = "parse into structured fields before encoding"
            else:
                action = ("target/frequency encoding or native categorical support; "
                          "one-hot will explode")
            flags.append({"severity": "medium", "column": c,
                          "issue": f"high cardinality ({col['n_unique']} levels)",
                          "action": action})
        if stype in ("latitude", "longitude") and not swap:
            valid_pct = col.get("coordinate_valid_pct")
            if valid_pct is not None and valid_pct < COORDINATE_VALID:
                lo, hi = ("-90..90", "latitude") if stype == "latitude" else ("-180..180", "longitude")
                flags.append({"severity": "medium", "column": c,
                              "issue": f"{stype} values outside valid {lo} range",
                              "action": f"validate units, swapped columns, and bad geocodes before modelling {hi}"})
        if stype in ("email", "url", "json_like") and col["role"] not in ("identifier", "text"):
            flags.append({"severity": "low", "column": c,
                          "issue": f"{stype} values detected",
                          "action": "extract structured features instead of treating raw strings as ordinary categories"})
        st = col.get("stats") or {}
        if col["role"] == "numeric" and stype not in ("latitude", "longitude"):
            skew = abs(st.get("skew") or 0)
            magnitude = col.get("magnitude") or 0
            if skew > SKEW_FLAG and magnitude >= EXTREME_RANGE:
                # a few rows a million times larger than the median decide a
                # linear fit on their own; that is a structural problem, not a nuance
                flags.append({"severity": "medium", "column": c,
                              "issue": f"skew {st['skew']} across {magnitude} orders of magnitude (max {st['max']})",
                              "action": "log/quantile transform or clip; a handful of extreme rows will otherwise "
                                        "dominate linear and distance-based fits"})
            elif skew > SKEW_FLAG:
                flags.append({"severity": "low", "column": c,
                              "issue": f"skew {st['skew']}",
                              "action": "log/quantile transform helps linear and distance-based models"})

    # leakage screen
    assoc = profile.get("relationships", {}).get("target_association", {})
    for c, a in assoc.items():
        if a["strength"] >= LEAKAGE_ASSOC:
            flags.append({"severity": "critical", "column": c,
                          "issue": f"association with target {a['strength']} ({a['method']})",
                          "action": "verify this is not leakage before using"})

    miss_assoc = profile.get("relationships", {}).get("missingness_association", {})
    for c, a in miss_assoc.items():
        sev = "critical" if a["strength"] >= LEAKAGE_ASSOC else "medium"
        note = ("whether this column is null predicts the target on its own, which "
                "usually means it is populated after the outcome is known"
                if sev == "critical" else
                "missingness carries signal; a null-indicator feature may help, but "
                "check the nulls are not caused by the outcome")
        flags.append({"severity": sev, "column": c,
                      "issue": f"missingness associated with target {a['strength']}",
                      "action": note})

    dup = profile["dataset"]["duplicate_rows_pct"]
    if dup and dup > 1:
        flags.append({"severity": "medium", "column": None,
                      "issue": f"{dup}% duplicate rows",
                      "action": "dedupe before splitting or the same row lands in train and test"})

    tgt = profile.get("target")
    if tgt:
        flags.extend(_target_flags(tgt))

    return flags


def _build_notes(profile, roles, semantic_types, target) -> dict:
    """Concrete preprocessing guidance derived from the profile."""
    cols = profile["columns"]
    numeric = [c for c, v in cols.items() if v["role"] in ("numeric", "discrete_numeric")]
    continuous_numeric = [
        c for c, v in cols.items()
        if v["role"] == "numeric" and semantic_types.get(c) not in ("latitude", "longitude")
    ]
    binary = [c for c, v in cols.items() if v["role"] == "binary"]
    categorical = [c for c, v in cols.items() if v["role"] in ("categorical", "binary")]
    highcard = [c for c in categorical if cols[c]["n_unique"] > HIGH_CARDINALITY]
    datetimes = [c for c, v in cols.items() if v["role"] == "datetime"]
    text = [c for c, v in cols.items() if v["role"] == "text"]
    coordinates = [c for c in cols if semantic_types.get(c) in ("latitude", "longitude")]
    structured_strings = [
        c for c in cols if semantic_types.get(c) in ("email", "url", "json_like", "address", "location")
    ]
    droppable = [c for c, v in cols.items() if v["role"] in ("constant", "empty", "identifier")]
    needs_impute = [c for c, v in cols.items() if (v.get("missing_pct") or 0) > 0]

    # do the numeric columns live on wildly different scales?
    mags = [cols[c].get("magnitude") for c in continuous_numeric if cols[c].get("magnitude") is not None]
    scale_spread = _r(max(mags) - min(mags), 1) if len(mags) > 1 else None

    n_rows = profile["dataset"]["n_rows"]
    n_feat = len(numeric) + len(categorical)
    preprocessing = {
        "drop": droppable,
        "impute": needs_impute,
        "scale": continuous_numeric if (scale_spread or 0) > 2 else [],
        "encode_low_cardinality": [c for c in categorical if c not in highcard],
        "encode_high_cardinality": highcard,
        "derive_datetime_parts": datetimes,
        "vectorize_or_embed_text": text,
        "validate_coordinates": coordinates,
        "parse_structured_strings": structured_strings,
    }

    notes = {
        "numeric_features": numeric,
        "continuous_numeric_features": continuous_numeric,
        "binary_features": binary,
        "categorical_features": categorical,
        "high_cardinality_features": highcard,
        "datetime_features": datetimes,
        "text_features": text,
        "coordinate_features": coordinates,
        "structured_string_features": structured_strings,
        "recommend_drop": droppable,
        "needs_imputation": needs_impute,
        "preprocessing_recommendations": preprocessing,
        "scale_spread_orders_of_magnitude": scale_spread,
        "scaling_required_for": (
            "linear models, SVM, KNN, neural nets"
            if (scale_spread or 0) > 2 else "probably not critical"
        ),
        "rows_per_feature": _r(n_rows / n_feat, 1) if n_feat else None,
    }

    tgt = profile.get("target")
    if tgt:
        notes["task_type"] = tgt["task_type"]
        if tgt["task_type"].endswith("classification"):
            notes["split_strategy"] = "stratified k-fold on the target"
            notes["baseline_to_beat"] = f"majority class = {tgt['majority_class_baseline']}% accuracy"
            notes["suggested_metrics"] = (
                ["pr_auc", "f1", "balanced_accuracy"]
                if (tgt.get("imbalance_ratio") or 1) > IMBALANCE_FLAG
                else ["roc_auc", "accuracy", "f1"]
            )
        else:
            st = tgt.get("stats") or {}
            skewed = abs(st.get("skew") or 0) > SKEW_FLAG
            notes["split_strategy"] = "random k-fold"
            notes["baseline_to_beat"] = "median predictor" if skewed else "mean predictor"
            notes["suggested_metrics"] = ["mae", "rmse", "r2"] if skewed else ["rmse", "mae", "r2"]
            if not skewed:
                notes["target_transform"] = "none needed"
            elif (st.get("negatives_pct") or 0) > 0:
                notes["target_transform"] = "quantile or yeo-johnson (skewed, has negatives; log1p undefined)"
            else:
                notes["target_transform"] = "consider log1p via TransformedTargetRegressor (skewed, non-negative)"
    else:
        notes["task_type"] = "no_target_provided"
        notes["split_strategy"] = "choose after selecting a supervised target, or use unsupervised validation"
        notes["suggested_metrics"] = []

    if datetimes:
        notes["split_warning"] = (
            f"datetime column(s) {datetimes} present: if rows are a time series, "
            "use a chronological split, not a random one"
        )
    if coordinates:
        notes["coordinate_guidance"] = (
            "validate ranges, then use raw coordinates for tree models or derive distance/geohash/"
            "region features for linear models"
        )
    if highcard:
        notes["high_cardinality_guidance"] = (
            "prefer native categorical support, frequency encoding, hashing, or target encoding "
            "inside cross-validation; avoid fitting target encoders before the split"
        )
    if text:
        notes["text_guidance"] = (
            "use TF-IDF, embeddings, or text-specific models; do not ordinal-encode free text"
        )
    if (notes.get("rows_per_feature") or 999) < 10:
        notes["capacity_warning"] = (
            "fewer than 10 rows per feature: prefer regularised linear models or "
            "shallow trees, and expect variance in the estimates"
        )

    notes["leakage_reminder"] = (
        "fit every transform (imputer, scaler, encoder) on the training fold only"
    )
    return notes


# ----------------------------------------------------------------------------
# text rendering
# ----------------------------------------------------------------------------

NOTE_ORDER = (
    "task_type", "split_strategy", "suggested_metrics", "baseline_to_beat", "target_transform",
    "scale_spread_orders_of_magnitude", "scaling_required_for", "rows_per_feature",
    "recommend_drop", "needs_imputation", "preprocessing_recommendations",
    "high_cardinality_guidance", "coordinate_guidance", "text_guidance", "split_warning",
    "capacity_warning", "leakage_reminder",
)


def _cell(x: Any) -> str:
    """A value that can sit in a markdown table cell without breaking the row."""
    return str(x).replace("|", "\\|").replace("\n", " ")


def render_profile(p: dict, max_columns: int = 60) -> str:
    """Compact markdown. This is what you put in an agent prompt.
    Args:
        p (dict): The profile data.
        max_columns (int): The maximum number of columns to display.
    Returns:
        str: The rendered profile.
    """
    L: list[str] = []
    d = p["dataset"]

    L.append("# DATASET PROFILE")
    L.append(
        f"{d['n_rows']:,} rows x {d['n_columns']} columns | {d['memory_mb']} MB | "
        f"{d['complete_rows_pct']}% complete rows | {d['duplicate_rows_pct']}% duplicates"
    )
    if d["values_redacted"]:
        L.append("(actual values redacted; structure only)")

    t = p.get("target")
    if t:
        L.append(f"\n## TARGET: {t['name']} -> {t['task_type']}")
        if t.get("description"):
            L.append(t["description"])
        if "class_distribution" in t:
            dist = ", ".join(f"{c['class']} {c['pct']}%" for c in t["class_distribution"][:8])
            L.append(f"classes ({t['n_unique']}): {dist}")
            L.append(f"imbalance {t.get('imbalance_ratio')}:1 | "
                     f"majority-class baseline {t['majority_class_baseline']}%")
        elif t.get("stats"):
            s = t["stats"]
            L.append(f"mean {s['mean']} | std {s['std']} | median {s['median']} | "
                     f"range [{s['min']}, {s['max']}] ({t.get('magnitude')} orders of magnitude) | "
                     f"skew {s['skew']} | zeros {s['zeros_pct']}% | negatives {s['negatives_pct']}%")

    L.append("\n## COLUMNS")
    L.append("| column | role | miss% | uniq | detail |")
    L.append("|---|---|---|---|---|")
    for i, (c, col) in enumerate(p["columns"].items()):
        if i >= max_columns:
            L.append(f"| ... | | | | {len(p['columns']) - max_columns} more columns omitted |")
            break
        st = col.get("stats") or {}
        role = col["role"]
        if col.get("semantic_type") and col["semantic_type"] != col["role"]:
            role = f"{col['role']}:{col['semantic_type']}"
        if col["role"] in ("numeric", "discrete_numeric"):
            detail = (f"med {st.get('median')} [{st.get('min')}, {st.get('max')}] "
                      f"skew {st.get('skew')} out {st.get('iqr_outlier_pct')}%")
            if col.get("coordinate_valid_pct") is not None:
                detail += f" valid_coord {col['coordinate_valid_pct']}%"
        elif col["role"] in ("categorical", "binary", "identifier"):
            tops = col.get("top_categories", [])[:3]
            detail = ", ".join(f"{_cell(x['value'])} {x['pct']}%" for x in tops)
            if col.get("rare_categories"):
                detail += f" (+{col['rare_categories']} rare)"
        elif col["role"] == "datetime":
            detail = (f"{st.get('min')} -> {st.get('max')} ({st.get('span_days')}d, "
                      f"parsed {st.get('parse_success_pct')}%)")
        elif col["role"] == "text":
            detail = f"~{st.get('mean_words')} words, max {st.get('max_chars')} chars"
        else:
            detail = ""
        if col.get("description"):
            detail = f"{_cell(col['description'])}; {detail}" if detail else _cell(col["description"])
        L.append(f"| {_cell(c)} | {role} | {col['missing_pct']} | {col['n_unique']} | {detail} |")

    assoc = p.get("relationships", {}).get("target_association")
    if assoc:
        L.append("\n## ASSOCIATION WITH TARGET (strongest first)")
        items = list(assoc.items())[:15]
        L.append(", ".join(f"{c} {a['strength']}" for c, a in items))
        L.append("(spearman for numeric-numeric, correlation ratio for mixed, cramers V for "
                 "categorical-categorical; all bias-corrected and scaled 0-1; identifiers excluded)")

    miss = p.get("relationships", {}).get("missingness_association")
    if miss:
        L.append("\n## MISSINGNESS ASSOCIATED WITH TARGET")
        L.append(", ".join(f"{c} {a['strength']}" for c, a in list(miss.items())[:10]))
        L.append("(strength of is-null indicator vs target; high values are a leakage warning)")

    red = p.get("relationships", {}).get("redundant_pairs")
    if red:
        L.append("\n## NEAR-DUPLICATE FEATURE PAIRS")
        L.append(", ".join(f"{r['a']}~{r['b']} ({r['spearman_abs']})" for r in red[:10]))

    if p["flags"]:
        L.append("\n## FLAGS")
        for f in p["flags"]:
            where = f"{f['column']}: " if f["column"] else ""
            L.append(f"- [{f['severity'].upper()}] {where}{f['issue']} -> {f['action']}")

    n = p["modelling_notes"]
    L.append("\n## MODELLING NOTES")
    for k in NOTE_ORDER:
        value = n.get(k)
        if value in (None, [], ""):
            continue
        if isinstance(value, dict):
            # one line per non-empty step, not a dict repr
            L.append(f"- {k}:")
            for step, columns in value.items():
                if columns:
                    L.append(f"  - {step}: {columns}")
        else:
            L.append(f"- {k}: {value}")

    return "\n".join(L)


def profile_to_json(p: dict, indent: int = 2) -> str:
    return json.dumps(p, indent=indent, default=str)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Profile a tabular CSV dataset for modelling.")
    parser.add_argument("path", nargs="?", default="data/train.csv", help="CSV file to profile")
    parser.add_argument("target", nargs="?", default=None, help="optional target column")
    parser.add_argument("--redact-values", action="store_true",
                        help="hide actual category labels and datetime values")
    parser.add_argument("--max-categories", type=int, default=10,
                        help="maximum category levels to list per column")
    parser.add_argument("--target-type", choices=["numeric", "categorical"], default=None,
                        help="what the target is, when the profiler should not guess")
    args = parser.parse_args()

    frame = pd.read_csv(args.path)
    print(render_profile(profile_dataset(
        frame,
        target=args.target,
        include_values=not args.redact_values,
        max_categories=args.max_categories,
        target_type=args.target_type,
    )))
