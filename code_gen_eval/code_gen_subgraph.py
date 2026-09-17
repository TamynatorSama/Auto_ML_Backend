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

from automl_runtime.reference import REFERENCE_CANDIDATE
from models import AttemptRecord, ModelResult, RunContext
from code_gen_eval.patching import EditError, resolve_reply
from code_gen_eval.runner import attempt_dir, run_candidate
from utils.prompts.code_gen_prompt import code_gen_prompt
from utils.prompts.code_fixer import code_fixer_prompt
from utils.prompts.code_judge import code_judge_prompt
from utils.reusable.eligibility import TIER_NAMES, assess, ranked_eligible, replay, used_excluded
from utils.reusable.guards import check_final
from utils.reusable.leakage import add_exclusions, columns_to_ablate, judge_ablation, read_exclusions
from utils.reusable.lessons import error_signature, raised, ran, read_lessons, record_lesson, render_lessons
from utils.reusable.llm import message_text
from utils.reusable.requirements import render_requirements, requirements_for

load_dotenv()

DEPENDENCY_RETRY_LIMIT = 1      # a missing package is not fixable by rewriting
FINAL_EVAL_CANDIDATES = 2       # winner, then runner-up, if the winner cannot be scored
MAX_REPAIRS = 2                 # repair rounds allowed per modelling attempt


def execution_cap(context: RunContext) -> int:
    """Candidate runs one model may use in total, repairs included.

    Repairs do not spend modelling attempts, so without a cap a model that never
    scores can run max_tries x (1 + MAX_REPAIRS) times.
    """
    return 2 * context.max_tries + 1

# statuses the fixer can do something about: the script is mechanically wrong.
# A clean timeout is not here on purpose — nothing is broken, the configuration
# is too expensive, and cutting its cost is a modelling decision for the judge.
# A run that raised and then hung comes back as "error", not "timeout".
BROKEN = ("error", "syntax_error")


def _broken(record: AttemptRecord) -> bool:
    """A failure the fixer can act on, and whose fix is a lesson for the run.

    A MemoryError raised inside the evaluator belongs here too: its traceback
    names the call that asked for too much, a dense one-hot or a pairwise
    matrix, and the repair is mechanical. A process stopped from outside for its
    size names nothing; making the model smaller is the judge's decision.
    """
    return record.status in BROKEN or (record.status == "out_of_memory" and raised(record))

# what every candidate has to contain to be worth evaluating; checked before an
# attempt is spent, and explained to the model in these words when it is not
CANDIDATE_CONTRACT = {
    "def build_pipeline": (
        "the harness calls build_pipeline(columns, task, ctx) and cross-validates the "
        "estimator it returns, so a module without it cannot be evaluated. A reply that "
        "stops after the header or says the rest is unchanged is not a complete module."
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
    extension_used: bool  # the one extra generation granted after a leak removed every usable attempt
    judge_notes: str
    status: str
    result: ModelResult


def _latest(attempts: List[AttemptRecord]) -> Optional[AttemptRecord]:
    """The newest modelling attempt; leak checks are measurements, not attempts."""
    return next((a for a in reversed(attempts) if a.kind != "ablation"), None)


def _executions(attempts: List[AttemptRecord]) -> int:
    """Candidate runs spent on modelling and repair; leak checks are the harness's, not the model's."""
    return sum(1 for a in attempts if a.kind != "ablation" and a.status != "skipped")


def _stuck(record: AttemptRecord, attempts: List[AttemptRecord]) -> bool:
    """The last repair reproduced the error it was meant to fix, word for word."""
    signature = error_signature(record)
    if signature is None:
        return False
    earlier = [
        a for a in attempts
        if a.kind != "ablation" and a.attempt < record.attempt and a.generation == record.generation
    ]
    return bool(earlier) and error_signature(earlier[-1]) == signature


def _leak_checks(record: AttemptRecord, attempts: List[AttemptRecord]) -> List[AttemptRecord]:
    subject = record.cached_from or record.attempt
    return [a for a in attempts if a.kind == "ablation" and a.ablation_of == subject and a.verdict]


def _llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)


def _libraries(context: RunContext) -> str:
    """Installed libraries WITH versions: code written for an older API is the
    most common avoidable failure, and a name alone does not say which API."""
    installed = ", ".join(f"{name} {version}" for name, version in sorted(context.environment.items()))
    return f"{installed}, automl_runtime" if installed else "automl_runtime"


def _describe_attempt(
    record: AttemptRecord, context: RunContext, attempts: Optional[List[AttemptRecord]] = None
) -> str:
    """What one attempt did, as the next generator or the judge needs to read it."""
    lines = [f"attempt {record.attempt}: {record.status}"]
    if record.changes:
        lines.append(f"changes: {record.changes}")
    if record.cv_scores:
        lines.append(
            "cv (computed by the harness): "
            + ", ".join(f"{k} {v:.4f}" for k, v in record.cv_scores.items())
        )
    folds = [value for value in record.fold_scores if value is not None]
    if len(folds) > 1:
        lines.append(f"{context.primary_metric} by fold: " + ", ".join(f"{value:.4f}" for value in folds))
    if record.wall_seconds is not None:
        lines.append(f"took {record.wall_seconds}s of a {context.time_budget_seconds}s budget")
    for warning in record.warnings:
        lines.append(f"WARNING: {warning}")
    for check in _leak_checks(record, attempts or []):
        lines.append(f"LEAK CHECK ({check.verdict}): {check.decision}")
    if record.traceback:
        lines.append(f"traceback:\n{record.traceback[-1500:]}")
    elif record.stdout:
        lines.append(f"stdout tail:\n{record.stdout[-800:]}")
    return "\n".join(lines)


def _removed_columns(context: RunContext) -> str:
    """Columns the candidate will not receive, and why, so it never names them."""
    lines = [f"- {column}: dropped by the profile" for column in context.drop_columns]
    for column, evidence in read_exclusions(context.run_dir).items():
        if column not in context.drop_columns:
            lines.append(f"- {column}: {evidence.get('reason', 'excluded by a leak check')}")
    return "\n".join(lines) or "none"


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
        f"libraries available: {_libraries(context)}\n\n"
        f"## COLUMNS REMOVED FROM THIS RUN\n{_removed_columns(context)}\n\n"
        f"{_pitfalls(context)}"
        f"## REQUIRED PREPROCESSING FOR {model}\n"
        f"{render_requirements(requirements_for(context.preprocessing_requirements, model))}\n\n"
        f"{context.summary}"
    )


def _pitfalls(context: RunContext) -> str:
    rendered = render_lessons(read_lessons(context.run_dir))
    return f"{rendered}\n\n" if rendered else ""


def generate_code(state: CodeGenSubgraphState) -> CodeGenSubgraphState:
    context, model = state["context"], state["model"]
    attempt = state.get("attempt", 0) + 1
    current_code = state.get("current_code", "")
    history = state.get("attempts", [])

    # search/replace cannot reliably repair a file that does not parse: the
    # anchors sit inside the broken region, so the same error comes straight
    # back and the attempt is spent for nothing
    last = _latest(history)
    broken = last is not None and last.status == "syntax_error"

    if not current_code or broken:
        problem = f"\n\n## WHAT WENT WRONG\n{last.traceback}" if broken else ""
        reference = f"\n\n## LAST CANDIDATE (did not parse)\n{current_code}" if broken else ""
        human = (
            f"{_brief(context, model)}{problem}{reference}\n\n"
            "Reply with MODE: REWRITE and a complete candidate module."
        )
    else:
        generation = state.get("generation", 0) + 1
        human = (
            f"{_brief(context, model)}\n\n"
            f"## PREVIOUS ATTEMPT\n{_describe_attempt(last, context, history)}\n\n"
            f"## JUDGE\n{state.get('judge_notes', 'no notes')}\n\n"
            f"## CURRENT CODE\n{current_code}\n\n"
            f"This is generation {generation} of {context.max_tries}. Reply with MODE: EDIT and "
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
            reply = resolve_reply(message_text(response), current_code, CANDIDATE_CONTRACT)
            return reply.code, reply.changes
        except EditError as error:
            if correction == 1:
                break
            messages.append(HumanMessage(content=f"{error}\n\nTry again."))

    response = model.invoke(
        [
            SystemMessage(content=code_gen_prompt),
            HumanMessage(content=f"{human}\n\nReply with MODE: REWRITE and a complete candidate module."),
        ]
    )
    try:
        # the model is asked for a rewrite but often answers with edits again;
        # accept those against the current module rather than discard them
        reply = resolve_reply(message_text(response), current_code, CANDIDATE_CONTRACT)
    except EditError as error:
        if current_code:
            # keep the working module: re-evaluating identical code is a cache
            # hit that costs nothing, while an empty module spends an attempt
            # and hands the fixer nothing to fix
            return current_code, f"the generator's reply could not be applied ({str(error)[:160]}); module unchanged"
        # run_code records an attempt with no candidate rather than raising, so
        # the worker survives and the reason lands in the record
        return "", f"the generator returned no usable candidate: {error}"
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
            script_path=str(out_dir / "candidate.py"),
            status="error",
            changes=state.get("changes", ""),
            traceback=reason,
        )
        (out_dir / "record.json").write_text(record.model_dump_json(indent=2), encoding="utf-8")
        print(f"[{state['model']}] attempt {record.attempt}: {record.status} (no script) — {reason}")
        return {"attempts": [record]}

    record = run_candidate(
        state["current_code"],
        state["model"],
        state["attempt"],
        context,
        backend=context.backend,
        changes=state.get("changes", ""),
        prior_attempts=[r.attempt for r in state.get("attempts", []) if r.kind != "ablation"],
        generation=state.get("generation", 1),
        kind="repair" if state.get("repairs", 0) else "generate",
        # columns excluded by any worker's leak check so far
        extra_excluded=list(read_exclusions(context.run_dir)),
    )
    print(f"[{state['model']}] attempt {record.attempt}: {record.status} {record.cv_scores}")

    # an attempt that cleared the stage the previous one failed at, whether a
    # repair or a new generation after repairs ran out, is a fix every worker
    # should know. A repair that produced no module ran nothing, so the failure
    # being fixed is the last one a candidate actually ran into.
    failed = next((a for a in reversed(state.get("attempts", [])) if a.kind != "ablation" and ran(a)), None)
    if failed is not None and _broken(failed):
        lesson = record_lesson(context.run_dir, state["model"], failed, record)
        if lesson:
            print(f"[{state['model']}] lesson: {lesson['signature'][:100]} -> {lesson['fix'][:80]}")
    return {"attempts": [record]}


def fix_code(state: CodeGenSubgraphState) -> CodeGenSubgraphState:
    """Make the script run again, without changing what it is trying to do.

    Repairs are counted separately from modelling attempts because they are a
    different job: a model whose every attempt died on an import error never
    produced a score at all, while the loop recorded four honest tries.
    """
    context, model = state["context"], state["model"]
    record = state["attempts"][-1]

    if not state.get("current_code", "").strip():
        # nothing to repair: no module was produced. A fixer writing from
        # nothing cannot see the brief and guesses the interface, so the
        # generator, which has it, writes the module instead
        human = (
            f"{_brief(context, model)}\n\n"
            f"## WHAT WENT WRONG\n{record.traceback}\n\n"
            "Reply with MODE: REWRITE and a complete candidate module."
        )
        code, changes = _ask_for_code(human, "")
        print(f"[{model}] repair {state.get('repairs', 0) + 1}: regenerated from the brief")
        return {
            "attempt": state["attempt"] + 1,
            "repairs": state.get("repairs", 0) + 1,
            "current_code": code,
            "changes": f"repair: {changes}",
        }

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
        f"libraries available: {_libraries(context)}\n\n"
        f"{_pitfalls(context)}"
        f"## WHAT HAPPENED\n{_describe_attempt(record, context)}{already}\n\n"
        f"## CANDIDATE\n{state['current_code']}"
    )

    llm = _llm()
    reason = "the fixer produced no reply"
    messages = [SystemMessage(content=code_fixer_prompt), HumanMessage(content=human)]
    for correction in range(2):
        response = llm.invoke(messages)
        try:
            reply = resolve_reply(message_text(response), state["current_code"], CANDIDATE_CONTRACT)
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
    """A broken candidate goes to the fixer, a suspiciously good one to a leak
    check, and a scored one to the judge."""
    record = state["attempts"][-1]
    attempts = state["attempts"]
    if (
        _broken(record)
        and state.get("repairs", 0) < MAX_REPAIRS
        # a repair that changed nothing about the error will not fix it on a second go
        and not _stuck(record, attempts)
        and _executions(attempts) < execution_cap(state["context"])
    ):
        return "fix_code"
    suspicious = any(finding.kind == "suspect_leakage" for finding in record.findings)
    if record.status in ("ok", "cached") and suspicious and not _leak_checks(record, state["attempts"]):
        return "verify"
    return "judge"


def verify(state: CodeGenSubgraphState) -> CodeGenSubgraphState:
    """Settle a suspiciously good attempt with a measurement, not an opinion.

    Re-run the same candidate on the same folds without the columns carrying most
    of its importance. When it cannot run without them, make the comparison with
    the harness's reference model instead. The leak is confirmed when most of the
    gain over the baseline disappears; the verdict is recorded on the last
    measurement, and a confirmed leak on columns the schema does not declare
    available is excluded for every model under the default policy.
    """
    context, model = state["context"], state["model"]
    attempts = state["attempts"]
    record = _latest(attempts)
    metric = context.primary_metric
    floor = context.baseline.cv_scores.get(metric) if context.baseline.applicable else None
    attempt = state["attempt"]
    excluded_so_far = list(read_exclusions(context.run_dir))

    suspects = columns_to_ablate(Path(record.script_path).parent)
    declared = [column for column in suspects if column in context.available_at_prediction]
    columns = [column for column in suspects if column not in declared]
    measurements: List[AttemptRecord] = []

    def measure(code: str, extra: List[str], label: str) -> AttemptRecord:
        nonlocal attempt
        attempt += 1
        measured = run_candidate(
            code, model, attempt, context, backend=context.backend, changes=label,
            generation=record.generation, kind="ablation", extra_excluded=extra,
            ablation_of=record.cached_from or record.attempt,
        )
        measurements.append(measured)
        return measured

    if not suspects or not columns:
        verdict = "declared_available" if declared else "unresolved"
        decision = (
            f"its score depends most on {', '.join(declared)}, which the schema declares known at "
            "prediction time, so they are legitimate inputs"
            if declared
            else "no feature importance to choose columns from, so the suspicion stands unverified"
        )
        full = ablated = retained = None
    else:
        source = "the candidate"
        full = record.cv_scores.get(metric)
        without = measure(
            state["current_code"], [*excluded_so_far, *columns],
            f"leak check: attempt {record.attempt} without {', '.join(columns)}",
        )
        ablated = without.cv_scores.get(metric) if without.status == "ok" else None

        if ablated is None:
            # the candidate names those columns and cannot run without them
            source = "the reference model"
            with_all = measure(REFERENCE_CANDIDATE, excluded_so_far, "leak check: reference model, all columns")
            without = measure(
                REFERENCE_CANDIDATE, [*excluded_so_far, *columns],
                f"leak check: reference model without {', '.join(columns)}",
            )
            full = with_all.cv_scores.get(metric) if with_all.status == "ok" else None
            ablated = without.cv_scores.get(metric) if without.status == "ok" else None

        confirmed, retained = judge_ablation(full, ablated, floor, context.metric_direction)
        named = ", ".join(columns)
        if confirmed is None:
            verdict = "unresolved"
            decision = f"the comparison without {named} could not be measured, so the suspicion stands unverified"
        else:
            shift = f"{metric} {full:.4g} -> {ablated:.4g}"
            if not confirmed:
                verdict = "cleared"
                decision = f"without {named}, {source} kept {retained:.0%} of its gain over the baseline ({shift}); not a leak"
            elif context.leakage_policy == "exclude":
                verdict = "confirmed"
                decision = (
                    f"without {named}, {source} kept only {retained:.0%} of its gain over the baseline "
                    f"({shift}); leak confirmed, so {named} are excluded from the run"
                )
                add_exclusions(context.run_dir, columns, {
                    "reason": f"leak confirmed by {model} attempt {record.attempt}: {decision}",
                    "found_by": model,
                    "attempt": record.attempt,
                    "metric": metric,
                    "score_with": full,
                    "score_without": ablated,
                    "retained_gain": retained,
                    "measured_with": source,
                })
            else:
                verdict = "kept_by_policy"
                decision = (
                    f"without {named}, {source} kept only {retained:.0%} of its gain ({shift}); a leak by the "
                    "numbers, but the run's leakage policy keeps them"
                )

    if not measurements:
        # nothing was run, but the verdict still needs a record the rules can find
        measurements.append(AttemptRecord(
            model=model, attempt=record.attempt, generation=record.generation, kind="ablation",
            script_path=record.script_path, status="skipped", ablation_of=record.cached_from or record.attempt,
        ))
    check = measurements[-1]
    check.verdict, check.decision, check.ablated_columns = verdict, decision, columns
    if check.status != "skipped":
        (Path(check.script_path).parent / "record.json").write_text(check.model_dump_json(indent=2), encoding="utf-8")

    print(f"[{model}] leak check ({verdict}): {decision}")
    return {"attempt": attempt, "attempts": measurements}


def judge(state: CodeGenSubgraphState) -> CodeGenSubgraphState:
    context = state["context"]
    attempts = state["attempts"]
    record = _latest(attempts)

    # a traceback, a timeout or a missing package is its own instruction; the
    # generator is already told to fix exactly that, so this saves an LLM call
    if record.status != "ok":
        return {"judge_notes": f"the attempt did not produce a score ({record.status}); fix that first"}

    earlier = [a for a in attempts if a.kind != "ablation" and a.attempt != record.attempt]
    history = "\n\n".join(_describe_attempt(r, context, attempts) for r in earlier)
    human = (
        f"{_brief(context, state['model'])}\n\n"
        f"## THIS ATTEMPT\n{_describe_attempt(record, context, attempts)}\n\n"
        f"## CANDIDATE\n{state['current_code']}\n\n"
        f"## EARLIER ATTEMPTS\n{history or 'none'}"
    )

    response = _llm().invoke(
        [SystemMessage(content=code_judge_prompt), HumanMessage(content=human)]
    )
    notes = message_text(response).strip()
    print(f"[{state['model']}] {notes.splitlines()[0][:120]}")
    return {"judge_notes": notes}


def decide(state: CodeGenSubgraphState) -> CodeGenSubgraphState:
    """Deterministic. No LLM decides whether a model keeps going.

    The best attempt and the patience count are replayed from the whole history
    against current eligibility, so an attempt blocked after the fact (a leak
    confirmed here or by another worker) stops counting, and the honest attempt
    that replaced it is judged on its own terms instead of against a score it
    was never supposed to match.
    """
    context = state["context"]
    attempts = state.get("attempts", [])
    record = _latest(attempts)
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

    # errors, timeouts and blocked attempts leave patience alone: they are
    # mechanical or validity failures the next attempt can fix, not stagnation.
    # A cached attempt counts: identical code is a no-op, and the loop can spin.
    exclusions = read_exclusions(context.run_dir)
    best_attempt, best_score, patience = replay(attempts, context, exclusions)
    extension_used = state.get("extension_used", False)

    if _executions(attempts) >= execution_cap(context):
        status = "execution_cap"
        print(f"[{state['model']}] stopped at {_executions(attempts)} candidate runs (cap {execution_cap(context)})")
    elif state.get("generation", 1) >= context.max_tries:
        # only an attempt that scored on a column excluded since earns the extra
        # generation. Checking that exclusions merely exist gave one to a model
        # that never beat the baseline and one that never scored at all.
        lost_to_exclusion = any(
            used_excluded(record, exclusions) for record in attempts
            if record.kind != "ablation" and record.status in ("ok", "cached") and record.cv_scores
        )
        if best_attempt is None and lost_to_exclusion and not extension_used:
            # every attempt that scored used a column excluded since; one more
            # generation builds without it rather than ending with nothing
            status, extension_used = "running", True
            print(f"[{state['model']}] every scored attempt used an excluded column; one more generation")
        else:
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
        "extension_used": extension_used,
    }


def route(state: CodeGenSubgraphState) -> str:
    return "generate_code" if state["status"] == "running" else "final_eval"


def final_eval(state: CodeGenSubgraphState) -> CodeGenSubgraphState:
    """Score the winning attempt on the test set, once, after the loop is over."""
    context, model = state["context"], state["model"]
    attempts = state.get("attempts", [])
    # read again here: another worker may have excluded a column since decide ran
    exclusions = read_exclusions(context.run_dir)
    ranked = ranked_eligible(attempts, context, exclusions)

    if not ranked:
        status = "unavailable" if state.get("status") == "unavailable" else "failed"
        blocked = [
            (record, assess(record, attempts, exclusions)[1])
            for record in attempts
            if record.kind != "ablation" and record.status in ("ok", "cached") and record.cv_scores
        ]
        if blocked:
            error = "every scored attempt was blocked: " + "; ".join(
                f"attempt {record.attempt}: {reason}" for record, reason in blocked[:3]
            )
            return {"result": ModelResult(
                model=model, status=status, attempts=attempts, error=error[:1000],
                eligibility="blocked", eligibility_note=blocked[0][1],
            )}
        last = _latest(attempts)
        return {"result": ModelResult(
            model=model, status=status, attempts=attempts,
            error=last.traceback[:1000] if last else "no attempt ran",
        )}

    # the best selectable attempt first, then the next. Final mode fits on every
    # training row and predicts the test file, which the loop never does, so a
    # failure only shows up here, and scoring the runner-up beats losing the model
    notes = []
    for tier, candidate in ranked[:FINAL_EVAL_CANDIDATES]:
        code = Path(candidate.script_path).read_text(encoding="utf-8")
        scored = run_candidate(
            code, model, candidate.attempt, context,
            backend=context.backend, final=True, changes=candidate.changes,
            generation=candidate.generation, kind=candidate.kind,
            extra_excluded=candidate.excluded_columns,
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
                    test_warnings=scored.warnings + check_final(candidate, scored, context),
                    error="; ".join(notes),
                    eligibility=TIER_NAMES[tier],
                    eligibility_note=assess(candidate, attempts, exclusions)[1],
                )
            }

        note = (
            f"attempt {candidate.attempt} could not be scored on the test set ({scored.status}): "
            f"{(scored.traceback or 'no test scores were produced')[:400]}"
        )
        print(f"[{model}] {note}")
        notes.append(note)

    tier, winner = ranked[0]
    return {
        "result": ModelResult(
            model=model,
            status=state.get("status", "max_tries"),
            attempts=attempts,
            best_attempt=winner.attempt,
            best_cv_scores=winner.cv_scores,
            error="; ".join(notes) or "no attempt could be scored on the test set",
            eligibility=TIER_NAMES[tier],
        )
    }


graph = StateGraph(CodeGenSubgraphState)
graph.add_node("generate_code", generate_code)
graph.add_node("run_code", run_code)
graph.add_node("fix_code", fix_code)
graph.add_node("verify", verify)
graph.add_node("judge", judge)
graph.add_node("decide", decide)
graph.add_node("final_eval", final_eval)

graph.add_edge(START, "generate_code")
graph.add_edge("generate_code", "run_code")
graph.add_conditional_edges("run_code", route_after_run, ["fix_code", "verify", "judge"])
graph.add_edge("fix_code", "run_code")
graph.add_edge("verify", "judge")
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
