"""
requirements.py
---------------
Turn the dataset profile into preprocessing the generated script MUST do.

Some preprocessing is not a modelling choice, it is hygiene, and leaving it to
chance is what made the unstable models unstable. Across one session's runs on
the same data:

    random_forest  31.4 - 35.7 mae     (scale invariant, robust to outliers)
    lightgbm       38.7 - 43.4 mae
    ridge          60.7 - 277.1 mae    (destroyed by both)
    mlp            51.3 - failed

Ridge did not vary because ridge is unpredictable. It varied because each run
rewrote the script from scratch, and whether it clipped a 254,545,454 outlier
and transformed a target with skew 17 was decided afresh every time. The models
that need the preprocessing to be right are exactly the ones that swung.

So the decisions are made once, here, from the numbers in the profile, and
handed to every script as requirements. Nothing is hardcoded about this
dataset: a clean target produces no target requirement, a well-behaved column
produces none of its own.

    derive_requirements(profile) -> list[PreprocessingRequirement]
"""

from __future__ import annotations

from typing import List

from models import PreprocessingRequirement

# families whose fit is dominated by feature scale and extreme values. Trees and
# boosters split on order and do not care, so requiring a transform of them adds
# risk without benefit.
SCALE_SENSITIVE = (
    "ridge", "lasso", "elastic_net", "linear_regression", "logistic_regression",
    "svm", "knn", "mlp",
)
ALL_MODELS = ("*",)

SKEW_FLAG = 1.0             # |skew| above this distorts a linear fit
HEAVY_SKEW = 3.0            # and above this it dominates the fit entirely
SCALE_SPREAD_FLAG = 2.0     # orders of magnitude between numeric columns
HIGH_CARDINALITY = 50       # categories above which one-hot and native handling both hurt
OUTLIER_FLAG = 5.0          # iqr outlier percentage worth naming


def _numeric_columns(profile: dict) -> dict:
    return {
        name: column
        for name, column in profile["columns"].items()
        if column["role"] in ("numeric", "discrete_numeric")
    }


def derive_requirements(profile: dict) -> List[PreprocessingRequirement]:
    requirements: List[PreprocessingRequirement] = []
    notes = profile.get("modelling_notes", {})
    target = profile.get("target") or {}

    # --- the target -------------------------------------------------------
    stats = target.get("stats") or {}
    skew = stats.get("skew")
    if target.get("task_type") == "regression" and skew is not None and abs(skew) > SKEW_FLAG:
        minimum = stats.get("min")
        transform = (
            "TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1)"
            if minimum is not None and minimum > -1
            else "TransformedTargetRegressor with a transform valid for negative values, "
                 "such as QuantileTransformer(output_distribution='normal')"
        )
        requirements.append(
            PreprocessingRequirement(
                column=target.get("name"),
                issue=f"target skew is {skew}, so a few large values dominate the fit",
                requirement=(
                    f"fit through {transform} from sklearn.compose, so predictions and "
                    "every reported score stay in the target's original units"
                ),
                applies_to=list(SCALE_SENSITIVE),
            )
        )

    # --- individual numeric columns --------------------------------------
    for name, column in _numeric_columns(profile).items():
        column_stats = column.get("stats") or {}
        column_skew = column_stats.get("skew")
        outliers = column_stats.get("iqr_outlier_pct")

        if column_skew is not None and abs(column_skew) > HEAVY_SKEW:
            requirements.append(
                PreprocessingRequirement(
                    column=name,
                    issue=(
                        f"{name} has skew {column_skew} and a range of "
                        f"[{column_stats.get('min')}, {column_stats.get('max')}]"
                    ),
                    requirement=(
                        "clip it to a high percentile or pass it through "
                        "QuantileTransformer(output_distribution='normal') inside the "
                        "pipeline before fitting"
                    ),
                    applies_to=list(SCALE_SENSITIVE),
                )
            )
        elif outliers is not None and outliers > OUTLIER_FLAG:
            requirements.append(
                PreprocessingRequirement(
                    column=name,
                    issue=f"{name} has {outliers}% of rows outside the interquartile fence",
                    requirement="clip or robust-scale it inside the pipeline",
                    applies_to=list(SCALE_SENSITIVE),
                )
            )

        valid = column.get("coordinate_valid_pct")
        if valid is not None and valid < 100:
            requirements.append(
                PreprocessingRequirement(
                    column=name,
                    issue=f"{name} has {round(100 - valid, 2)}% of values outside its valid range",
                    requirement=(
                        "clip to the valid range or treat the invalid rows as missing, "
                        "as a pipeline step"
                    ),
                    applies_to=list(ALL_MODELS),
                )
            )

    # --- everything together ---------------------------------------------
    spread = notes.get("scale_spread_orders_of_magnitude")
    if spread is not None and spread > SCALE_SPREAD_FLAG:
        requirements.append(
            PreprocessingRequirement(
                column=None,
                issue=f"numeric columns span {spread} orders of magnitude",
                requirement="scale the numeric columns inside the pipeline",
                applies_to=list(SCALE_SENSITIVE),
            )
        )

    for name, column in profile["columns"].items():
        variants = column.get("label_variants")
        if not variants:
            continue
        example = " / ".join(repr(label) for label in variants["examples"][0])
        requirements.append(
            PreprocessingRequirement(
                column=name,
                issue=(
                    f"{name} writes {variants['groups']} categories more than one way, differing only "
                    f"by case or spacing (e.g. {example})"
                ),
                requirement=(
                    "pass it through automl_runtime.CategoryCleaner before encoding, so each "
                    "category becomes one column rather than several"
                ),
                applies_to=list(ALL_MODELS),
            )
        )

    for name in notes.get("high_cardinality_features", []) or []:
        levels = profile["columns"][name]["n_unique"]
        requirements.append(
            PreprocessingRequirement(
                column=name,
                issue=f"{name} has {levels} categories",
                requirement=(
                    "frequency or target encode it inside the pipeline. Do not one-hot it "
                    "and do not hand it to a booster as a native categorical: both make the "
                    "fit enormous"
                ),
                applies_to=list(ALL_MODELS),
            )
        )

    return requirements


def requirements_for(requirements: List[PreprocessingRequirement], model: str) -> List[PreprocessingRequirement]:
    return [r for r in requirements if "*" in r.applies_to or model in r.applies_to]


def render_requirements(requirements: List[PreprocessingRequirement]) -> str:
    if not requirements:
        return "none for this model"
    return "\n".join(
        f"- {r.issue}\n  -> {r.requirement}" for r in requirements
    )
