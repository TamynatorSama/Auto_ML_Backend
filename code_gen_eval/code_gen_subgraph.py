import sys
from pathlib import Path

# Allow imports from project root when running this file directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Annotated, List, Optional, TypedDict
import operator

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

from models import AttemptRecord, ModelResult, RunContext
from code_gen_eval.patching import EditError, resolve_reply
from code_gen_eval.runner import RESULT_SENTINEL, attempt_dir, run_script
from utils.prompts.code_gen_prompt import code_gen_prompt
from utils.prompts.code_fixer import code_fixer_prompt
from utils.prompts.code_judge import code_judge_prompt
from utils.reusable.llm import message_text
from utils.reusable.metrics import is_improvement
from utils.reusable.requirements import render_requirements, requirements_for

load_dotenv()

DEPENDENCY_RETRY_LIMIT = 1      # a missing package is not fixable by rewriting
FINAL_EVAL_CANDIDATES = 2       # winner, then runner-up, if the winner cannot be scored
MAX_REPAIRS = 2                 # repair rounds allowed per modelling attempt

# statuses the fixer can do something about: the script is mechanically wrong.
# A clean timeout is not here on purpose — nothing is broken, the configuration
# is too expensive, and cutting its cost is a modelling decision for the judge.
# A run that raised and then hung comes back as "error", not "timeout".
BROKEN = ("error", "syntax_error")

# what every script has to contain to be worth running; checked before an
# attempt is spent, and explained to the model in these words when it is not
SCRIPT_CONTRACT = {
    RESULT_SENTINEL: (
        "the runner reads the scores from the JSON printed after this line, so a "
        "script without it cannot be scored. A reply that stops after the header "
        "or says the rest is unchanged is not a complete script."
    ),
}


class CodeGenSubgraphState(TypedDict, total=False):
    # handed over in the Send payload, never written by this graph
    context: RunContext
    model: str

    # private to one worker. The parent declares none of these, so parallel
    # workers cannot collide on them.
    attempt: int          # every execution, so artifact directories stay unique
    generation: int       # modelling attempts only; repairs do not spend these
    repairs: int
    current_code: str
    changes: str
    attempts: Annotated[List[AttemptRecord], operator.add]
    best_attempt: Optional[int]
    best_score: Optional[float]
    patience: int
    dependency_retries: int
    judge_notes: str
    status: str
    result: ModelResult


def _llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)


def _describe_attempt(record: AttemptRecord, context: RunContext) -> str:
    """What one attempt did, as the next generator or the judge needs to read it."""
    lines = [f"attempt {record.attempt}: {record.status}"]
    if record.changes:
        lines.append(f"changes: {record.changes}")
    if record.cv_scores:
        lines.append("cv: " + ", ".join(f"{k} {v:.4f}" for k, v in record.cv_scores.items()))
    if record.wall_seconds is not None:
        lines.append(f"took {record.wall_seconds}s of a {context.time_budget_seconds}s budget")
    for warning in record.warnings:
        lines.append(f"WARNING: {warning}")
    if record.traceback:
        lines.append(f"traceback:\n{record.traceback[-1500:]}")
    elif record.stdout:
        lines.append(f"stdout tail:\n{record.stdout[-800:]}")
    return "\n".join(lines)


def _brief(context: RunContext, model: str) -> str:
    baseline = context.baseline
    floor = (
        ", ".join(f"{k} {v:.4f}" for k, v in baseline.cv_scores.items())
        if baseline.applicable
        else "none"
    )
    return (
        f"model: {model}\n"
        f"task: {context.task_type} on target `{context.target}`\n"
        f"metrics: {', '.join(context.eval_matrics)} (first is primary, "
        f"{context.metric_direction} is better)\n"
        f"baseline to beat ({baseline.strategy}): {floor}\n"
        f"libraries available: {', '.join(sorted(context.environment))}\n\n"
        f"## REQUIRED PREPROCESSING FOR {model}\n"
        f"{render_requirements(requirements_for(context.preprocessing_requirements, model))}\n\n"
        f"{context.summary}"
    )


def generate_code(state: CodeGenSubgraphState) -> CodeGenSubgraphState:
    context, model = state["context"], state["model"]
    attempt = state.get("attempt", 0) + 1
    current_code = state.get("current_code", "")
    history = state.get("attempts", [])

    # search/replace cannot reliably repair a file that does not parse: the
    # anchors sit inside the broken region, so the same error comes straight
    # back and the attempt is spent for nothing
    broken = bool(history) and history[-1].status == "syntax_error"

    if not current_code or broken:
        problem = f"\n\n## WHAT WENT WRONG\n{history[-1].traceback}" if broken else ""
        reference = f"\n\n## LAST SCRIPT (did not parse)\n{current_code}" if broken else ""
        human = (
            f"{_brief(context, model)}{problem}{reference}\n\n"
            "Reply with MODE: REWRITE and a complete script."
        )
    else:
        last = history[-1]
        human = (
            f"{_brief(context, model)}\n\n"
            f"## PREVIOUS ATTEMPT\n{_describe_attempt(last, context)}\n\n"
            f"## JUDGE\n{state.get('judge_notes', 'no notes')}\n\n"
            f"## CURRENT CODE\n{current_code}\n\n"
            f"This is attempt {attempt} of {context.max_tries}. Reply with MODE: EDIT and "
            f"search/replace blocks against CURRENT CODE."
        )

    code, changes = _ask_for_code(human, "" if broken else current_code)
    return {
        "attempt": attempt,
        "generation": state.get("generation", 0) + 1,
        "repairs": 0,
        "current_code": code,
        "changes": changes,
    }


def _ask_for_code(human: str, current_code: str) -> tuple:
    """One generation, with a single corrective round trip if the edits do not apply.

    patching.EditError already explains what went wrong in words meant for the
    model, so handing the message straight back is usually enough. If it still
    fails, ask for the whole script rather than losing the attempt.

    A reply that parses but yields a script that does not parse, or one with no
    result sentinel, is caught here too: both are certain failures, and finding
    out by running them cost an attempt each time it happened.
    """
    model = _llm()
    messages = [SystemMessage(content=code_gen_prompt), HumanMessage(content=human)]

    for correction in range(2):
        response = model.invoke(messages)
        try:
            reply = resolve_reply(message_text(response), current_code, SCRIPT_CONTRACT)
            return reply.code, reply.changes
        except EditError as error:
            if correction == 1:
                break
            messages.append(HumanMessage(content=f"{error}\n\nTry again."))

    response = model.invoke(
        [
            SystemMessage(content=code_gen_prompt),
            HumanMessage(content=f"{human}\n\nReply with MODE: REWRITE and a complete script."),
        ]
    )
    try:
        reply = resolve_reply(message_text(response), "", SCRIPT_CONTRACT)
    except EditError as error:
        # run_code records an attempt with no script rather than raising, so
        # the worker survives and the reason lands in the record
        return "", f"the generator returned no usable script: {error}"
    return reply.code, reply.changes


def run_code(state: CodeGenSubgraphState) -> CodeGenSubgraphState:
    context = state["context"]

    if not state.get("current_code", "").strip():
        # the reason lives only here, so it is written to disk and printed like
        # any other attempt: a generation failure that leaves no artifact is one
        # nobody can diagnose afterwards
        out_dir = attempt_dir(context, state["model"], state["attempt"])
        out_dir.mkdir(parents=True, exist_ok=True)
        reason = state.get("changes") or "the generator returned no usable script"

        record = AttemptRecord(
            model=state["model"],
            attempt=state["attempt"],
            generation=state.get("generation", 1),
            kind="repair" if state.get("repairs", 0) else "generate",
            script_path=str(out_dir / "script.py"),
            status="error",
            changes=state.get("changes", ""),
            traceback=reason,
        )
        (out_dir / "record.json").write_text(record.model_dump_json(indent=2), encoding="utf-8")
        print(f"[{state['model']}] attempt {record.attempt}: {record.status} (no script) — {reason}")
        return {"attempts": [record]}

    record = run_script(
        state["current_code"],
        state["model"],
        state["attempt"],
        context,
        backend=context.backend,
        changes=state.get("changes", ""),
        prior_attempts=[r.attempt for r in state.get("attempts", [])],
        generation=state.get("generation", 1),
        kind="repair" if state.get("repairs", 0) else "generate",
    )
    print(f"[{state['model']}] attempt {record.attempt}: {record.status} {record.cv_scores}")
    return {"attempts": [record]}


def fix_code(state: CodeGenSubgraphState) -> CodeGenSubgraphState:
    """Make the script run again, without changing what it is trying to do.

    Repairs are counted separately from modelling attempts because they are a
    different job: a model whose every attempt died on an import error never
    produced a score at all, while the loop recorded four honest tries.
    """
    context, model = state["context"], state["model"]
    record = state["attempts"][-1]

    # what has already been tried on this same script. Without it the fixer
    # re-proposed a repair that had just failed, because all it could see was
    # the latest traceback.
    generation = state.get("generation", 1)
    tried = [
        r for r in state.get("attempts", [])[:-1]
        if r.generation == generation and r.kind == "repair"
    ]
    already = (
        "\n\n## REPAIRS ALREADY TRIED ON THIS SCRIPT\n"
        + "\n".join(
            f"- {r.changes or 'unrecorded'} -> still {r.status}" for r in tried
        )
        + "\nDo not propose any of these again. The same error after a repair means "
          "the cause is somewhere else."
        if tried else ""
    )

    human = (
        f"model: {model}\n"
        f"libraries available: {', '.join(sorted(context.environment))}\n\n"
        f"## WHAT HAPPENED\n{_describe_attempt(record, context)}{already}\n\n"
        f"## SCRIPT\n{state['current_code']}"
    )

    llm = _llm()
    reason = "the fixer produced no reply"
    messages = [SystemMessage(content=code_fixer_prompt), HumanMessage(content=human)]
    for correction in range(2):
        response = llm.invoke(messages)
        try:
            reply = resolve_reply(message_text(response), state["current_code"], SCRIPT_CONTRACT)
            print(f"[{model}] repair {state.get('repairs', 0) + 1}: {reply.changes[:90]}")
            return {
                "attempt": state["attempt"] + 1,
                "repairs": state.get("repairs", 0) + 1,
                "current_code": reply.code,
                "changes": f"repair: {reply.changes}",
            }
        except EditError as error:
            # python unbinds the exception variable once the block ends, so the
            # message has to be kept if it is wanted after the loop
            reason = str(error)
            if correction == 1:
                break
            messages.append(HumanMessage(content=f"{error}\n\nTry again."))

    return {
        "attempt": state["attempt"] + 1,
        "repairs": state.get("repairs", 0) + 1,
        "current_code": "",
        "changes": f"the fixer returned no usable script: {reason}",
    }


def route_after_run(state: CodeGenSubgraphState) -> str:
    """A broken script goes to the fixer; a scored one goes to the judge."""
    record = state["attempts"][-1]
    if record.status in BROKEN and state.get("repairs", 0) < MAX_REPAIRS:
        return "fix_code"
    return "judge"


def judge(state: CodeGenSubgraphState) -> CodeGenSubgraphState:
    context = state["context"]
    record = state["attempts"][-1]

    # a traceback, a timeout or a missing package is its own instruction; the
    # generator is already told to fix exactly that, so this saves an LLM call
    if record.status != "ok":
        return {"judge_notes": f"the attempt did not produce a score ({record.status}); fix that first"}

    history = "\n\n".join(_describe_attempt(r, context) for r in state["attempts"][:-1])
    human = (
        f"{_brief(context, state['model'])}\n\n"
        f"## THIS ATTEMPT\n{_describe_attempt(record, context)}\n\n"
        f"## CODE\n{state['current_code']}\n\n"
        f"## EARLIER ATTEMPTS\n{history or 'none'}"
    )

    response = _llm().invoke(
        [SystemMessage(content=code_judge_prompt), HumanMessage(content=human)]
    )
    notes = message_text(response).strip()
    print(f"[{state['model']}] {notes.splitlines()[0][:120]}")
    return {"judge_notes": notes}


def decide(state: CodeGenSubgraphState) -> CodeGenSubgraphState:
    """Deterministic. No LLM decides whether a model keeps going."""
    context = state["context"]
    record = state["attempts"][-1]
    metric = context.improvement_metric
    best_score = state.get("best_score")
    best_attempt = state.get("best_attempt")
    patience = state.get("patience", 0)
    dependency_retries = state.get("dependency_retries", 0)

    if record.status == "dependency_error":
        # one corrective try in case it was a wrong import path; a genuinely
        # absent package cannot be fixed by editing the script
        dependency_retries += 1
        if dependency_retries > DEPENDENCY_RETRY_LIMIT:
            status = "unavailable"
        elif state.get("generation", 1) >= context.max_tries:
            status = "max_tries"
        else:
            status = "running"
        return {
            "dependency_retries": dependency_retries,
            "status": status,
            "patience": patience,
        }

    # errors and timeouts spend an attempt but are not score stagnation, so
    # patience is left alone for them: a traceback is a mechanical failure the
    # next attempt can fix
    score = record.cv_scores.get(metric)
    if score is None:
        pass
    elif best_score is None or is_improvement(
        best_score, score, metric, context.improvement_delta, context.improvement_mode
    ):
        best_score, best_attempt, patience = score, record.attempt, 0
    else:
        # a cached attempt lands here too: identical code is a no-op, and
        # without counting it the loop can spin
        patience += 1

    if state.get("generation", 1) >= context.max_tries:
        status = "max_tries"
    elif patience >= context.early_stopping_patience:
        status = "no_improvement"
    else:
        status = "running"

    return {
        "best_score": best_score,
        "best_attempt": best_attempt,
        "patience": patience,
        "status": status,
        "dependency_retries": dependency_retries,
    }


def route(state: CodeGenSubgraphState) -> str:
    return "generate_code" if state["status"] == "running" else "final_eval"


def final_eval(state: CodeGenSubgraphState) -> CodeGenSubgraphState:
    """Score the winning attempt on the test set, once, after the loop is over."""
    context, model = state["context"], state["model"]
    attempts = state.get("attempts", [])
    best_attempt = state.get("best_attempt")

    if best_attempt is None:
        last = attempts[-1] if attempts else None
        status = "unavailable" if state.get("status") == "unavailable" else "failed"
        return {
            "result": ModelResult(
                model=model,
                status=status,
                attempts=attempts,
                error=last.traceback[:1000] if last else "no attempt ran",
            )
        }

    # best first, then the next best. The final branch of a script never runs
    # during the loop, so a typo in it is only discovered here — and losing a
    # model that cross-validated perfectly well over one unreached line is a
    # worse answer than scoring its runner-up.
    metric = context.improvement_metric
    ranked = sorted(
        (r for r in attempts if r.cv_scores.get(metric) is not None and r.script_path),
        key=lambda r: r.cv_scores[metric] * (-1 if context.metric_direction == "higher" else 1),
    )[:FINAL_EVAL_CANDIDATES]

    notes = []
    for candidate in ranked:
        code = Path(candidate.script_path).read_text(encoding="utf-8")
        scored = run_script(
            code, model, candidate.attempt, context,
            backend=context.backend, final=True, changes=candidate.changes,
            generation=candidate.generation, kind=candidate.kind,
        )
        print(f"[{model}] final: attempt {candidate.attempt} {scored.status} | "
              f"cv {candidate.cv_scores} -> test {scored.test_scores}")

        if scored.test_scores:
            if notes:
                notes.append(f"scored attempt {candidate.attempt} instead")
            return {
                "result": ModelResult(
                    model=model,
                    status=state.get("status", "max_tries"),
                    attempts=attempts,
                    best_attempt=candidate.attempt,
                    best_cv_scores=candidate.cv_scores,
                    test_scores=scored.test_scores,
                    test_warnings=scored.warnings,
                    error="; ".join(notes),
                )
            }

        note = (
            f"attempt {candidate.attempt} could not be scored on the test set ({scored.status}): "
            f"{(scored.traceback or 'no AUTOML_FINAL branch in the script')[:400]}"
        )
        print(f"[{model}] {note}")
        notes.append(note)

    winner = next(r for r in attempts if r.attempt == best_attempt)
    return {
        "result": ModelResult(
            model=model,
            status=state.get("status", "max_tries"),
            attempts=attempts,
            best_attempt=best_attempt,
            best_cv_scores=winner.cv_scores,
            error="; ".join(notes) or "no attempt could be scored on the test set",
        )
    }


graph = StateGraph(CodeGenSubgraphState)
graph.add_node("generate_code", generate_code)
graph.add_node("run_code", run_code)
graph.add_node("fix_code", fix_code)
graph.add_node("judge", judge)
graph.add_node("decide", decide)
graph.add_node("final_eval", final_eval)

graph.add_edge(START, "generate_code")
graph.add_edge("generate_code", "run_code")
graph.add_conditional_edges("run_code", route_after_run, ["fix_code", "judge"])
graph.add_edge("fix_code", "run_code")
graph.add_edge("judge", "decide")
graph.add_conditional_edges("decide", route, ["generate_code", "final_eval"])
graph.add_edge("final_eval", END)

app = graph.compile()


if __name__ == "__main__":
    from test_values import context

    final = app.invoke(
        {"context": context.model_copy(update={"max_tries": 2}), "model": "ridge"},
        {"recursion_limit": 50},
    )
    print(final["result"].model_dump_json(indent=2, exclude={"attempts"}))
