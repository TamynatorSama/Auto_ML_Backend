"""
folds.py
--------
The cross-validation folds, computed once per run and frozen on disk.

Every model in a run has to be scored on the same rows or their scores are not
comparable. Rebuilding a splitter inside each candidate, even from the same
seed, left that to chance; freezing the row indices removes it.

    make_splitter(strategy, n_splits, seed, n_rows) -> sklearn splitter
    build_folds(strategy, n_splits, seed, y, groups) -> list of (train_idx, test_idx)
    save_folds(path, folds) / load_folds(path)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit

Fold = Tuple[np.ndarray, np.ndarray]

STRATEGIES = ("kfold", "stratified_kfold", "group_kfold", "time_series_split")


def make_splitter(strategy: str, n_splits: int, seed: int, n_rows: int):
    folds = max(2, min(n_splits, n_rows))

    if strategy == "time_series_split":
        return TimeSeriesSplit(n_splits=folds)
    if strategy == "group_kfold":
        return GroupKFold(n_splits=folds)
    if strategy == "stratified_kfold":
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    if strategy == "kfold":
        return KFold(n_splits=folds, shuffle=True, random_state=seed)

    raise ValueError(f"unknown cv strategy: {strategy}; expected one of {', '.join(STRATEGIES)}")


def build_folds(
    strategy: str,
    n_splits: int,
    seed: int,
    y: Sequence,
    groups: Optional[Sequence] = None,
) -> List[Fold]:
    y = np.asarray(y)
    splitter = make_splitter(strategy, n_splits, seed, len(y))
    placeholder = np.zeros((len(y), 1))
    return [
        (np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64))
        for train, test in splitter.split(placeholder, y, groups)
    ]


def save_folds(path: str | Path, folds: List[Fold]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for number, (train, test) in enumerate(folds):
        arrays[f"train_{number}"] = train
        arrays[f"test_{number}"] = test
    np.savez_compressed(path, n_folds=np.array(len(folds)), **arrays)
    return path


def load_folds(path: str | Path) -> List[Fold]:
    with np.load(path) as stored:
        count = int(stored["n_folds"])
        return [(stored[f"train_{number}"], stored[f"test_{number}"]) for number in range(count)]
