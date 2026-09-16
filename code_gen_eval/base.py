import sys
from pathlib import Path

# Allow imports from project root when running this file directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import TypedDict, Annotated, List, Optional
import math
import operator
import os

from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from models import Configs, ModelResult, RunContext, RunReport
from utils.reusable.report import build_report
from utils.reusable.report_html import render_html
from code_gen_eval.code_gen_subgraph import app as code_gen_subgraph

load_dotenv()


class CodeGenEvalState(TypedDict):
    topic: str
    context: RunContext
    config: Configs
    plan: List[str]
    # every worker writes its one ModelResult here in parallel
    completed_sections: Annotated[list, operator.add]
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
    """
    model = payload["model"]
    try:
        final = code_gen_subgraph.invoke(payload, {"recursion_limit": 100})
        return {"completed_sections": [final["result"]]}
    except Exception as error:
        print(f"[{model}] worker failed: {error!r}")
        return {"completed_sections": [ModelResult(model=model, status="failed", error=repr(error))]}


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
    """Best first by the primary metric, failures last.

    Workers finish in whatever order they finish, and `operator.add` appends in
    completion order, so nothing downstream may rely on list position. Test
    scores rank when present; a run that never reached final evaluation falls
    back to its cross-validation score.
    """
    metric = context.primary_metric
    worse_is_larger = context.metric_direction == "higher"

    def key(result: ModelResult):
        score = result.test_scores.get(metric, result.best_cv_scores.get(metric))
        if score is not None:
            return (0, -score if worse_is_larger else score, result.model)
        # tried and failed still ranks above never tried at all: the first is a
        # result, the second is an absence
        return (2 if result.status == "unavailable" else 1, 0.0, result.model)

    return sorted(results, key=key)


def summarize_results(state: CodeGenEvalState) -> CodeGenEvalState:
    context = state["context"]
    metric = context.primary_metric
    ranked = rank_results(state["completed_sections"], context)

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
        # the final evaluation re-runs the winning script, so its cross-validation
        # warnings are raised a second time; the reader needs each one once
        messages = list(dict.fromkeys(messages))
        notes = len(messages)
        if messages:
            flagged.append((result.model, messages))

        lines.append(
            f"| {position} | {result.model} | {result.status} | "
            f"{_fmt(result.test_scores.get(metric))} | "
            f"{_fmt(result.best_cv_scores.get(metric))} | {len(result.attempts)} | "
            f"{notes or ''} |"
        )

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
    structured = build_report(state["completed_sections"], context)
    out_dir = Path(context.run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(structured.model_dump_json(indent=2), encoding="utf-8")
    print(f"report: {out_dir / 'report.json'}")

    # the rendered page is a convenience for reading a run without a frontend,
    # not the deliverable; the JSON above is what an interface consumes
    if os.environ.get("AUTOML_HTML_REPORT") == "1":
        (out_dir / "report.html").write_text(render_html(structured), encoding="utf-8")

    return {"report": report, "run_report": structured}


graph = StateGraph(CodeGenEvalState)

graph.add_node("create_plan", create_plan)
graph.add_node("code_gen_worker", code_gen_worker)
graph.add_node("summarize_results", summarize_results)

graph.add_edge(START, "create_plan")
graph.add_conditional_edges("create_plan", assign_workers, ["code_gen_worker"])
graph.add_edge("code_gen_worker", "summarize_results")
graph.add_edge("summarize_results", END)

app = graph.compile()


if __name__ == "__main__":
    from test_values import context, config

    app.invoke(
        {"topic": "House prices prediction", "context": context, "config": config},
        {"max_concurrency": 2},
    )
