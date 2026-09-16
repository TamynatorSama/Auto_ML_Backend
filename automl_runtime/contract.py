"""
contract.py
-----------
Everything the evaluator needs to run one candidate, frozen into a file.

One contract is written into each attempt directory, so an attempt can be
re-run and audited on its own. It deliberately holds no path to the test set:
during the improvement loop there is nothing a candidate could read it from.
The test file is handed to the evaluator only in final mode.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Contract:
    model: str
    target: str
    task_type: str
    metrics: List[str]
    primary_metric: str
    train_path: str
    folds_path: str
    seed: int
    n_jobs: int
    time_budget_seconds: int
    excluded_columns: List[str] = field(default_factory=list)
    columns: List[dict] = field(default_factory=list)
    ordered: bool = False                  # rows are in time order (time_series_split)
    group_column: Optional[str] = None     # set only when folds are grouped

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Contract":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {key: value for key, value in data.items() if key in cls.__dataclass_fields__}
        return cls(**known)
