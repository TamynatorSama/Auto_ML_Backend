import sys
from pathlib import Path

# Allow imports from project root when running this file directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime
from typing import Dict, TypedDict, Annotated, List, Optional
import math
import operator
import os
import threading

from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from models import Configs, ModelResult, RunContext, RunReport
from utils.reusable.eligibility import BLOCKED, assess
from utils.reusable.leakage import read_exclusions
from utils.reusable.report import build_report, result_group
from utils.reusable.report_html import render_html
from code_gen_eval.code_gen_subgraph import app as code_gen_subgraph
from code_gen_eval.runner import release_run
from code_gen_eval.sandbox_client import SandboxUnavailable
from utils.reusable import hooks
from utils.reusable.hooks import RunStopped

load_dotenv()


class CodeGenEvalState(TypedDict):
    topic: str
    context: RunContext
    config: Configs
    plan: List[str]
    # every worker writes its one ModelResult here in parallel; a model re-run
    # after a late exclusion appends a second, and the latest one counts
    completed_sections: Annotated[list, operator.add]
    reruns: Annotated[list, operator.add]
    rerun_due: List[str]
    report: str
    run_report: RunReport


def create_plan(state: CodeGenEvalState) -> CodeGenEvalState:
    # the plan is already decided upstream, so this stays deterministic:
    # no planner LLM, no structured output, same fan-out every run
    unavailable = state["context"].unavailable_models
    return {
        "plan": list(state["config"].models),
        # a model nothing can import was never evaluated, which is a different
        # fact from one that was evaluated and lost; the report has to say which
        "completed_sections": [
            ModelResult(model=model, status="unavailable", error=reason)
            for model, reason in sorted(unavailable.items())
        ],
    }


RESULT_FILE = "result.json"
_slots: Dict[str, threading.BoundedSemaphore] = {}
_slots_lock = threading.Lock()


def _slot(context: RunContext) -> threading.BoundedSemaphore:
    """One limit per run on how many models train at once.

    Enforced inside the graph, from the run's resource plan, so any caller gets
    it: main.py invokes this graph with no concurrency setting at all, and two
    boosters on 306,000 rows at once ran the machine out of memory.
    """
    with _slots_lock:
        if context.run_dir not in _slots:
            _slots[context.run_dir] = threading.BoundedSemaphore(max(1, context.max_concurrency))
        return _slots[context.run_dir]


def _result_path(context: RunContext, model: str) -> Path:
    return Path(context.run_dir) / model / RESULT_FILE


def assign_workers(state: CodeGenEvalState):
    return [
        Send("code_gen_worker", {"context": state["context"], "model": model})
        for model in state["plan"]
    ]


def code_gen_worker(payload: dict) -> dict:
    """Run one model's generate/run/judge loop and return its one ModelResult.

    The subgraph is invoked here rather than added as a node so that a worker
    dying takes only itself down: an exception inside a Send branch otherwise
    kills the whole superstep, and one model running out of memory would lose
    the other four along with it.

    A model whose result is already saved in this run's directory is not run
    again: that is how a crashed run resumes. Its directory is set aside first
    when the model is being re-run on purpose, or when a crash left it half done.
    """
    context, model = payload["context"], payload["model"]
    rerun = bool(payload.get("rerun"))
    saved = _result_path(context, model)

    if saved.exists() and not rerun:
        try:
            result = ModelResult.model_validate_json(saved.read_text(encoding="utf-8"))
            print(f"[{model}] resumed: finished in an earlier session ({result.status})")
            hooks.emit(context.run_id, "model_resumed", **_summary(result))
            return {"completed_sections": [result]}
        except ValueError:
            pass

    hooks.emit(context.run_id, "model_waiting", model=model)
    try:
        hooks.check_stop(context.run_id)
        with _slot(context):
            hooks.check_stop(context.run_id)  # the stop may have come while this model waited
            hooks.emit(context.run_id, "model_started", model=model)
            model_dir = saved.parent
            if model_dir.exists():
                label = "before-rerun" if rerun else "interrupted"
                model_dir.rename(model_dir.with_name(f"{model}.{label}-{datetime.now():%Y%m%d%H%M%S}"))
            try:
                final = code_gen_subgraph.invoke({"context": context, "model": model}, {"recursion_limit": 150})
                result = final["result"]
            except (RunStopped, SandboxUnavailable):
                raise
            except Exception as error:
                print(f"[{model}] worker failed: {error!r}")
                result = ModelResult(model=model, status="failed", error=repr(error))
    except (RunStopped, SandboxUnavailable) as error:
        # not the model's failure: nothing is saved, so a resume trains it again
        print(f"[{model}] stopped: {error}")
        result = ModelResult(model=model, status="stopped", error=f"stopped: {error}")
        hooks.emit(context.run_id, "model_finished", **_summary(result))
        return {"completed_sections": [result]}

    saved.parent.mkdir(parents=True, exist_ok=True)
    saved.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    hooks.emit(context.run_id, "model_finished", **_summary(result))
    return {"completed_sections": [result], "reruns": [model] if rerun else []}


def _summary(result: ModelResult) -> dict:
    return {
        "model": result.model, "status": result.status, "best_attempt": result.best_attempt,
        "best_cv_scores": result.best_cv_scores, "test_scores": result.test_scores,
        "eligibility": result.eligibility, "error": result.error[:500],
    }


def latest_results(results: List[ModelResult]) -> List[ModelResult]:
    """One result per model: a re-run replaces the result it was run to replace."""
    latest: Dict[str, ModelResult] = {}
    for result in results:
        latest[result.model] = result
    return list(latest.values())


def review_results(state: CodeGenEvalState) -> CodeGenEvalState:
    """Find models blocked only by a leak another model confirmed after they finished.

    Their attempts all used a column the run has since excluded, so they have no
    selectable result through no fault of their own. Each is re-run once, now
    without the column, rather than reported as blocked with nothing in its place.
    """
    context = state["context"]
    results = latest_results(state["completed_sections"])
    recheck_eligibility(results, context)
    already = set(state.get("reruns") or [])
    due = [
        result.model for result in results
        if result.eligibility == "blocked"
        and "excluded from the run since" in result.eligibility_note
        and result.model not in already
    ]
    for model in due:
        print(f"[{model}] re-running without the columns excluded after it finished")
    return {"rerun_due": due}


def route_review(state: CodeGenEvalState):
    due = state.get("rerun_due") or []
    if not due:
        return "summarize_results"
    return [Send("code_gen_worker", {"context": state["context"], "model": model, "rerun": True}) for model in due]


def _fmt(value: Optional[float]) -> str:
    """A score, short enough to sit in a table.

    A diverged model can report 2e+289, and `:.4f` renders that as 290 digits
    that push every other column off the screen.
    """
    if value is None:
        return "-"
    if not math.isfinite(value):
        return "inf" if value > 0 else ("-inf" if value < 0 else "nan")
    if abs(value) >= 1e6 or (value != 0 and abs(value) < 1e-3):
        return f"{value:.3e}"
    return f"{value:.4f}"


def rank_results(results: List[ModelResult], context: RunContext) -> List[ModelResult]:
    """Selectable before unselectable, then best first by cross-validated score.

    Workers finish in whatever order they finish, and `operator.add` appends in
    completion order, so nothing downstream may rely on list position. Models are
    compared on cross-validation, not on the test set: the test score estimates
    how a chosen model generalises, and choosing by it would spend that estimate.
    """
    metric = context.primary_metric
    worse_is_larger = context.metric_direction == "higher"

    def key(result: ModelResult):
        score = result.best_cv_scores.get(metric)
        group = result_group(result, score)
        signed = 0.0 if score is None else (-score if worse_is_larger else score)
        return (group, signed, result.model)

    return sorted(results, key=key)


def recheck_eligibility(results: List[ModelResult], context: RunContext) -> None:
    """Block a finished model whose winner used a column excluded after it finished.

    Workers run in parallel, so one may complete before another confirms a leak
    in a column it used. The final exclusions decide, not the ones each worker
    happened to see.
    """
    exclusions = read_exclusions(context.run_dir)
    if not exclusions:
        return
    for result in results:
        winner = next((r for r in result.attempts if r.attempt == result.best_attempt), None)
        if winner is None or result.eligibility == "blocked":
            continue
        tier, reason = assess(winner, result.attempts, exclusions)
        if tier == BLOCKED:
            result.eligibility, result.eligibility_note = "blocked", reason


def summarize_results(state: CodeGenEvalState) -> CodeGenEvalState:
    context = state["context"]
    metric = context.primary_metric
    # the run is over: let go of what it held, in this process and on the sandbox host
    with _slots_lock:
        _slots.pop(context.run_dir, None)
    release_run(context)
    results = latest_results(state["completed_sections"])
    recheck_eligibility(results, context)
    ranked = rank_results(results, context)

    lines = [
        f"# Run {context.run_id}",
        "",
        f"{context.task_type} on `{context.target}` | primary metric **{metric}** "
        f"({context.metric_direction} is better)",
        f"split: {context.split_plan.method}, test_size {context.split_plan.test_size}, "
        f"seed {context.split_plan.random_seed} | cv: {context.split_plan.cv_strategy} "
        f"x{context.split_plan.cv_folds}",
        "",
        "| # | model | status | test | cv | attempts | warnings |",
        "|---|---|---|---|---|---|---|",
    ]

    flagged = []
    for position, result in enumerate(ranked, start=1):
        winner = next(
            (r for r in result.attempts if r.attempt == result.best_attempt), None
        )
        # the cross-validation warnings on the attempt that was chosen, the
        # warnings raised while scoring the test set, and any reason there is no
        # test score at all. All three existed in the data and none reached here.
        messages = list(winner.warnings) if winner else []
        messages += list(result.test_warnings)
        if result.error:
            messages.append(result.error)
        if result.eligibility != "clean" and result.eligibility_note:
            messages.append(f"{result.eligibility}: {result.eligibility_note}")
        # the final evaluation re-runs the winning script, so its cross-validation
        # warnings are raised a second time; the reader needs each one once
        messages = list(dict.fromkeys(messages))
        notes = len(messages)
        if messages:
            flagged.append((result.model, messages))

        status = result.status if result.eligibility == "clean" else f"{result.status} ({result.eligibility})"
        lines.append(
            f"| {position} | {result.model} | {status} | "
            f"{_fmt(result.test_scores.get(metric))} | "
            f"{_fmt(result.best_cv_scores.get(metric))} | {len(result.attempts)} | "
            f"{notes or ''} |"
        )

    exclusions = read_exclusions(context.run_dir)
    if exclusions:
        lines.append("")
        lines.append("## columns excluded during the run")
        for column, evidence in exclusions.items():
            lines.append(f"- **{column}**: {evidence.get('reason', 'excluded by a leak check')}")

    if flagged:
        lines.append("")
        lines.append("## warnings")
        for model, warnings in flagged:
            lines.append(f"- **{model}**")
            for warning in warnings:
                lines.append(f"  - {warning}")

    if context.environment:
        lines.append("")
        lines.append(
            "environment: " + ", ".join(f"{k} {v}" for k, v in sorted(context.environment.items()))
        )
    for note in context.environment_notes:
        lines.append(f"- {note}")

    baseline = context.baseline
    floor = baseline.cv_scores.get(metric)
    lines.append("")
    lines.append(
        f"baseline ({baseline.strategy}): {'not applicable' if not baseline.applicable else f'{metric} {floor:.4f} out-of-fold'}"
    )

    report = "\n".join(lines)
    print(report)

    # the console table is the short version. The structured report carries the
    # protocol, the per-attempt trace, the error breakdown and the artifacts,
    # which is what makes a number on a leaderboard worth believing.
    structured = build_report(results, context)
    out_dir = Path(context.run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(structured.model_dump_json(indent=2), encoding="utf-8")
    print(f"report: {out_dir / 'report.json'}")
    hooks.emit(context.run_id, "report_ready", status=structured.status, selected_model=structured.selected_model,
               path=str(out_dir / "report.json"))

    # the rendered page is a convenience for reading a run without a frontend,
    # not the deliverable; the JSON above is what an interface consumes
    if os.environ.get("AUTOML_HTML_REPORT") == "1":
        # a convenience must never cost the run: report.json is already written
        try:
            (out_dir / "report.html").write_text(render_html(structured), encoding="utf-8")
        except Exception as error:
            print(f"report.html was not written: {error!r}")

    return {"report": report, "run_report": structured}


graph = StateGraph(CodeGenEvalState)

graph.add_node("create_plan", create_plan)
graph.add_node("code_gen_worker", code_gen_worker)
graph.add_node("review_results", review_results)
graph.add_node("summarize_results", summarize_results)

graph.add_edge(START, "create_plan")
graph.add_conditional_edges("create_plan", assign_workers, ["code_gen_worker"])
graph.add_edge("code_gen_worker", "review_results")
graph.add_conditional_edges("review_results", route_review, ["code_gen_worker", "summarize_results"])
graph.add_edge("summarize_results", END)

app = graph.compile()


if __name__ == "__main__":
    from test_values import context, config

    app.invoke(
        {"topic": "House prices prediction", "context": context, "config": config},
        {"max_concurrency": 2},
    )
