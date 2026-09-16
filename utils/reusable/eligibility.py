"""
eligibility.py
--------------
Which attempts may be selected, and the best of them.

Selection used to know one thing: a better score wins. That rule threw away a
leak fix, because the attempt that exploited the leak scored better than the
one that removed it. Validity now comes first and score second:

    CLEAN       nothing stands against it
    UNVERIFIED  suspiciously good, and the leak check could not settle it
    BLOCKED     its predictions mean something else, it was confirmed to exploit
                a leak, or it used a column the run has since excluded

A blocked attempt is never selected. An unverified one is selected only when
no clean attempt exists. Eligibility is recomputed from the whole history every
time, so a column excluded late, even by another model's worker, retroactively
blocks every attempt that used it.

    assess(record, attempts, exclusions)      -> (tier, reason)
    replay(attempts, context, exclusions)     -> (best_attempt, best_score, patience)
    ranked_eligible(attempts, context, exclusions) -> list of (tier, record)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from models import AttemptRecord, RunContext
from utils.reusable.metrics import is_improvement

CLEAN, UNVERIFIED, BLOCKED = 0, 1, 2
TIER_NAMES = {CLEAN: "clean", UNVERIFIED: "unverified", BLOCKED: "blocked"}
SCORED = ("ok", "cached")
CLEARING_VERDICTS = ("cleared", "declared_available", "kept_by_policy")


def _leak_check(record: AttemptRecord, attempts: List[AttemptRecord]) -> Optional[AttemptRecord]:
    subject = record.cached_from or record.attempt
    checks = [a for a in attempts if a.kind == "ablation" and a.ablation_of == subject and a.verdict]
    return checks[-1] if checks else None


def assess(record: AttemptRecord, attempts: List[AttemptRecord], exclusions: Dict[str, dict]) -> Tuple[int, str]:
    blocking = [f for f in record.findings if f.severity == "blocking"]
    if blocking:
        return BLOCKED, blocking[0].message

    used = sorted(column for column in exclusions if column not in record.excluded_columns)
    if used:
        reason = exclusions[used[0]].get("reason", "excluded by a leak check")
        return BLOCKED, f"trained on {', '.join(used)}, excluded from the run since: {reason}"

    if any(f.kind == "suspect_leakage" for f in record.findings):
        check = _leak_check(record, attempts)
        if check is None or check.verdict == "unresolved":
            return UNVERIFIED, "suspiciously good and the leak check could not settle it"
        if check.verdict == "confirmed":
            return BLOCKED, f"a leak check confirmed it depends on {', '.join(check.ablated_columns)}"
        if check.verdict in CLEARING_VERDICTS:
            return CLEAN, ""
    return CLEAN, ""


def _candidates(attempts: List[AttemptRecord], context: RunContext) -> List[AttemptRecord]:
    metric = context.improvement_metric
    return [
        a for a in sorted(attempts, key=lambda a: a.attempt)
        if a.kind != "ablation" and a.status in SCORED and a.cv_scores.get(metric) is not None
    ]


def replay(
    attempts: List[AttemptRecord], context: RunContext, exclusions: Dict[str, dict]
) -> Tuple[Optional[int], Optional[float], int]:
    """The best selectable attempt and the patience count, replayed from the history.

    An attempt with a better tier replaces the best whatever its score; within a
    tier it must clear the improvement delta. Failed and blocked attempts leave
    patience alone: they are mechanical or validity failures, not stagnation.
    """
    metric = context.improvement_metric
    best: Optional[AttemptRecord] = None
    best_tier = BLOCKED
    patience = 0

    for record in _candidates(attempts, context):
        tier, _ = assess(record, attempts, exclusions)
        if tier == BLOCKED:
            continue
        score = record.cv_scores[metric]
        if best is None or tier < best_tier or (
            tier == best_tier and is_improvement(
                best.cv_scores[metric], score, metric, context.improvement_delta, context.improvement_mode
            )
        ):
            best, best_tier, patience = record, tier, 0
        else:
            patience += 1

    if best is None:
        return None, None, patience
    return best.attempt, best.cv_scores[metric], patience


def ranked_eligible(
    attempts: List[AttemptRecord], context: RunContext, exclusions: Dict[str, dict]
) -> List[Tuple[int, AttemptRecord]]:
    """Selectable attempts, best first: tier, then score."""
    metric = context.improvement_metric
    sign = 1 if context.metric_direction == "lower" else -1
    tiers = [(assess(record, attempts, exclusions)[0], record) for record in _candidates(attempts, context)]
    return sorted(
        ((tier, record) for tier, record in tiers if tier != BLOCKED),
        key=lambda pair: (pair[0], sign * pair[1].cv_scores[metric]),
    )
