import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from utils.reusable import summary
from utils.reusable.summary import profile_dataset, render_profile

rng = np.random.default_rng(0)
N = 4000


def flags_of(profile, severity=None):
    return [
        (f["severity"], f["column"], f["issue"])
        for f in profile["flags"]
        if severity is None or f["severity"] == severity
    ]


@pytest.fixture
def housing_like():
    sqft = rng.integers(300, 5000, N)
    return pd.DataFrame({
        "listing_id": np.arange(N),
        "sqft": sqft,
        "city": rng.choice(["Pune", "Delhi", "Mumbai", "Goa"], N),
        "posted_by": rng.choice(["Owner", "Dealer"], N),
        "price": (sqft * 137 + rng.integers(0, 50_000, N)).astype(int),
    })


# --- target role -----------------------------------------------------------------

def test_near_unique_integer_target_is_regression(housing_like):
    p = profile_dataset(housing_like, target="price")
    assert p["target"]["role"] == "numeric"
    assert p["target"]["task_type"] == "regression"
    assert p["modelling_notes"]["task_type"] == "regression"


def test_near_unique_integer_feature_is_still_an_identifier(housing_like):
    p = profile_dataset(housing_like, target="price")
    assert p["columns"]["listing_id"]["role"] == "identifier"
    assert "listing_id" in p["modelling_notes"]["recommend_drop"]


def test_few_level_integer_target_defaults_to_classification():
    df = pd.DataFrame({"x": rng.normal(size=N), "rating": rng.integers(1, 6, N)})
    p = profile_dataset(df, target="rating")
    assert p["target"]["task_type"] == "multiclass_classification"
    assert any("treated as classification" in issue for _, _, issue in flags_of(p, "low"))


def test_declared_numeric_target_regresses_few_level_integers():
    df = pd.DataFrame({"x": rng.normal(size=N), "rating": rng.integers(1, 6, N)})
    p = profile_dataset(df, target="rating", target_type="numeric")
    assert p["target"]["task_type"] == "regression"
    assert p["target"]["declared_type"] == "numeric"
    assert not any("treated as classification" in issue for _, _, issue in flags_of(p))


def test_declared_categorical_target_classifies_integer_codes():
    df = pd.DataFrame({"x": rng.normal(size=N), "code": rng.integers(0, 40, N)})
    p = profile_dataset(df, target="code", target_type="categorical")
    assert p["target"]["task_type"] == "multiclass_classification"


def test_degenerate_targets_are_critical():
    df = pd.DataFrame({"x": rng.normal(size=N), "const": 1, "uid": [f"u{i}" for i in range(N)]})
    assert any(col == "const" for _, col, _ in flags_of(profile_dataset(df, target="const"), "critical"))
    assert any(col == "uid" for _, col, _ in flags_of(profile_dataset(df, target="uid"), "critical"))


# --- association / leakage screen ------------------------------------------------

def test_identifier_feature_is_not_a_leak():
    # 45 jobs over 45 rows: one level per row explains any target perfectly
    df = pd.DataFrame({
        "job": [f"job {i}" for i in range(45)],
        "education": rng.integers(8, 20, 45),
        "salary": rng.integers(20_000, 90_000, 45),
    })
    p = profile_dataset(df, target="salary")
    assert "job" not in p["relationships"]["target_association"]
    assert flags_of(p, "critical") == []


def test_high_cardinality_noise_is_not_a_leak():
    # 1000 groups of 4 rows, no signal: raw eta is inflated, corrected is small
    df = pd.DataFrame({"group": [f"g{i // 4}" for i in range(N)], "y": rng.normal(size=N)})
    p = profile_dataset(df, target="y")
    assert p["relationships"]["target_association"]["group"]["strength"] < 0.3
    assert flags_of(p, "critical") == []


def test_real_leak_is_still_caught():
    df = pd.DataFrame({"x": rng.normal(size=N)})
    df["y"] = df["x"] * 3 + 1
    df["copy_of_y"] = df["y"] + rng.normal(scale=1e-3, size=N)
    p = profile_dataset(df, target="y")
    assert any(col == "copy_of_y" for _, col, _ in flags_of(p, "critical"))


def test_fast_cramers_v_matches_crosstab_formula():
    a = pd.Series(rng.choice(list("ABCDE"), N))
    b = pd.Series(rng.choice(list("xyz"), N))
    b[a == "A"] = "x"
    a.iloc[:20] = np.nan

    tab = pd.crosstab(a, b).to_numpy(dtype=float)
    n = tab.sum()
    expected = np.outer(tab.sum(1), tab.sum(0)) / n
    chi2 = ((tab - expected) ** 2 / expected).sum()
    r, k = tab.shape
    phi2corr = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    denom = min(k - (k - 1) ** 2 / (n - 1) - 1, r - (r - 1) ** 2 / (n - 1) - 1)
    reference = np.sqrt(phi2corr / denom)

    assert summary._cramers_v(a, b) == pytest.approx(reference, rel=1e-9)


def test_correlation_ratio_is_symmetric_in_missing_handling():
    cats = pd.Series(rng.choice(list("ABC"), N))
    vals = pd.Series(rng.normal(size=N)) + (cats == "A") * 2
    vals.iloc[:10] = np.nan
    cats.iloc[10:20] = None
    strength = summary._correlation_ratio(cats, vals)
    assert strength is not None and 0.5 < strength < 0.8


def test_oversized_contingency_is_skipped_not_built():
    a = pd.Series([f"a{i}" for i in range(3000)])
    b = pd.Series([f"b{i}" for i in range(3000)])
    assert summary._contingency(a, b) is None


# --- flags ------------------------------------------------------------------------

def test_swapped_coordinates_are_flagged():
    lat_actual = rng.uniform(8, 35, N)        # india-ish latitudes
    lon_actual = rng.uniform(68, 97, N)
    lon_actual[:40] = rng.uniform(100, 150, 40)   # a few rows that cannot be latitudes
    df = pd.DataFrame({"LATITUDE": lon_actual, "LONGITUDE": lat_actual, "y": rng.normal(size=N)})
    p = profile_dataset(df, target="y")
    issues = [issue for _, _, issue in flags_of(p, "medium")]
    assert any("may be swapped" in issue for issue in issues)
    assert not any("outside valid" in issue for issue in issues)   # not reported twice


def test_correct_coordinates_are_not_flagged():
    df = pd.DataFrame({"lat": rng.uniform(8, 35, N), "lon": rng.uniform(68, 97, N), "y": rng.normal(size=N)})
    assert not any("swapped" in issue for _, _, issue in flags_of(profile_dataset(df, target="y")))


def test_skewed_positive_target_suggests_log1p():
    df = pd.DataFrame({"x": rng.normal(size=N), "y": rng.lognormal(mean=4, sigma=1.5, size=N)})
    p = profile_dataset(df, target="y")
    flag = next(f for f in p["flags"] if f["column"] == "y" and "target skew" in f["issue"])
    assert "log1p" in flag["action"]
    assert "log1p" in p["modelling_notes"]["target_transform"]
    assert p["modelling_notes"]["baseline_to_beat"] == "median predictor"
    assert p["modelling_notes"]["suggested_metrics"][0] == "mae"


def test_skewed_target_with_negatives_does_not_suggest_log1p():
    y = rng.lognormal(mean=4, sigma=1.5, size=N)
    y[:100] = -y[:100]
    df = pd.DataFrame({"x": rng.normal(size=N), "y": y})
    p = profile_dataset(df, target="y")
    flag = next(f for f in p["flags"] if f["column"] == "y" and "target skew" in f["issue"])
    assert "log1p is undefined" in flag["action"]
    assert "yeo-johnson" in p["modelling_notes"]["target_transform"]


def test_symmetric_target_has_no_transform_note():
    df = pd.DataFrame({"x": rng.normal(size=N), "y": rng.normal(size=N)})
    p = profile_dataset(df, target="y")
    assert p["modelling_notes"]["target_transform"] == "none needed"
    assert not any("target skew" in issue for _, _, issue in flags_of(p))


def test_extreme_range_numeric_is_medium_not_low():
    area = rng.lognormal(mean=7, sigma=0.5, size=N)
    area[:5] = 2.5e8
    df = pd.DataFrame({"area": area, "y": rng.normal(size=N)})
    p = profile_dataset(df, target="y")
    flag = next(f for f in p["flags"] if f["column"] == "area")
    assert flag["severity"] == "medium" and "orders of magnitude" in flag["issue"]


# --- rendering --------------------------------------------------------------------

def test_render_escapes_pipes_and_carries_descriptions(housing_like):
    p = profile_dataset(
        housing_like, target="price",
        schema={"city": "locality | city", "price": "sale price in rupees"},
    )
    text = render_profile(p)
    row = next(line for line in text.splitlines() if line.startswith("| city |"))
    unescaped_pipes = len(re.findall(r"(?<!\\)\|", row))
    assert unescaped_pipes == 6                       # five cells, the pipe in the description escaped
    assert "locality \\| city" in row
    assert "sale price in rupees" in text
    assert "orders of magnitude" in text
    assert "- preprocessing_recommendations:" in text
    assert "{'drop'" not in text


def test_render_has_no_target_for_unsupervised():
    df = pd.DataFrame({"a": rng.normal(size=N), "b": rng.choice(list("xy"), N)})
    text = render_profile(profile_dataset(df))
    assert "## TARGET" not in text and "no_target_provided" in text


# --- speed guard ------------------------------------------------------------------

def test_high_cardinality_association_is_fast():
    import time
    n = 200_000
    df = pd.DataFrame({
        "addr": rng.choice([f"street {j}" for j in range(30_000)], n),
        "y": rng.integers(0, 2, n),
    })
    started = time.perf_counter()
    profile_dataset(df, target="y")
    assert time.perf_counter() - started < 5
