"""
splitting.py
------------
Execute a SplitPlan once, deterministically, so every model in the run is
scored on the exact same test rows.

The LLM decides WHAT the split should be (SplitPlan). This module decides HOW
it is carried out, in plain pandas with a fixed seed, so the same plan on the
same file always produces the same rows.

    apply_split_plan(df, plan)                 -> (train, test)
    write_split(train, test, output_dir)       -> (train_path, test_path)
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from models import SplitPlan


def _random_split(df: pd.DataFrame, plan: SplitPlan) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(plan.random_seed)
    order = rng.permutation(len(df)) if plan.shuffle else np.arange(len(df))
    n_test = int(round(len(df) * plan.test_size))
    return df.iloc[order[n_test:]], df.iloc[order[:n_test]]


def _stratified_split(df: pd.DataFrame, plan: SplitPlan) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(plan.random_seed)
    test_mask = np.zeros(len(df), dtype=bool)
    # one shuffled slice per class, so class proportions survive the cut
    for _, class_positions in sorted(df.groupby(plan.stratify_column, sort=True).indices.items()):
        taken = rng.permutation(class_positions)
        test_mask[taken[: int(round(len(taken) * plan.test_size))]] = True
    return df.iloc[~test_mask], df.iloc[test_mask]


def _temporal_split(df: pd.DataFrame, plan: SplitPlan) -> Tuple[pd.DataFrame, pd.DataFrame]:
    keys = pd.to_datetime(df[plan.time_column], errors="coerce")
    if keys.isna().all():
        keys = df[plan.time_column]
    # mergesort is stable: rows sharing a timestamp keep their file order
    order = np.argsort(keys.to_numpy(), kind="mergesort")
    n_test = int(round(len(df) * plan.test_size))
    return df.iloc[order[: len(df) - n_test]], df.iloc[order[len(df) - n_test :]]


def _grouped_split(df: pd.DataFrame, plan: SplitPlan) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(plan.random_seed)
    sizes = df.groupby(plan.group_column, sort=True).size()
    groups = rng.permutation(sizes.index.to_numpy())
    # whole groups go to test until the row budget is met, so no entity straddles the cut
    budget = len(df) * plan.test_size
    test_groups: list = []
    taken = 0
    for group in groups:
        if taken >= budget:
            break
        test_groups.append(group)
        taken += sizes[group]
    test_mask = df[plan.group_column].isin(test_groups).to_numpy()
    return df.iloc[~test_mask], df.iloc[test_mask]


_METHODS = {
    "random": _random_split,
    "stratified": _stratified_split,
    "temporal": _temporal_split,
    "grouped": _grouped_split,
}


def apply_split_plan(df: pd.DataFrame, plan: SplitPlan) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cut one frame into (train, test) exactly as the plan describes."""
    if plan.method not in _METHODS:
        raise ValueError(f"unknown split method: {plan.method}")

    if plan.drop_duplicates:
        df = df.drop_duplicates()

    df = df.reset_index(drop=True)
    train, test = _METHODS[plan.method](df, plan)

    if len(train) == 0 or len(test) == 0:
        raise ValueError(f"{plan.method} split produced an empty side: {len(train)} train, {len(test)} test")

    return train, test


def write_split(train: pd.DataFrame, test: pd.DataFrame, output_dir: str) -> Tuple[str, str]:
    """Persist the split so every model reads the same two files."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    train_path = directory / "train.csv"
    test_path = directory / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    return str(train_path), str(test_path)
