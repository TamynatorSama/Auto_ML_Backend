import sys
from pathlib import Path

# Allow imports from project root when running this file directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime
import hashlib
import os
from typing import TypedDict, Annotated, Any, Dict, List, Sequence, Tuple
import json

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt

from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from dotenv import load_dotenv
import pandas as pd

from models import (
    BaselineResult,
    Configs,
    DataSchema,
    DataType,
    PreprocessingRequirement,
    RunContext,
    SplitPlan,
)
from automl_runtime.folds import build_folds, load_folds, save_folds
from utils.reusable.leakage import add_exclusions, relation_screen, render_screen
from utils.reusable.summary import profile_dataset, render_profile
from utils.reusable.splitting import METHODS as SPLIT_METHODS, apply_split_plan, write_split
from utils.reusable.baseline import run_baseline
from utils.reusable.requirements import derive_requirements
from utils.reusable import hooks
from utils.reusable.resources import plan_resources
from utils.reusable.llm import message_text
from utils.reusable.metrics import METRIC_DIRECTION, REGRESSION_METRICS
from utils.reusable.environment import (
    ALL_MODELS,
    PROVISION_NEVER,
    PROVISION_ONCE,
    available_model_names,
    probe_environment,
    provision,
    resolve_models,
)
from utils.prompts.config_generator import config_generator_prompt
from utils.prompts.split_planner import split_planner_prompt

load_dotenv()


class InformationState(TypedDict):
    full_summary: str
    summary: str
    schema: DataSchema
    data_path: str
    train_path: str
    test_path: str
    target: str
    task_type: str
    drop_columns: List[str]
    preprocessing_requirements: List[PreprocessingRequirement]
    backend: str
    provision_policy: str
    environment: Dict[str, str]
    unavailable_models: Dict[str, str]
    environment_notes: List[str]
    config: Configs
    split_plan: SplitPlan
    baseline: BaselineResult
    context: RunContext
    # set by the caller to place the run; otherwise assigned in prepare_run
    run_id: int
    run_dir: str
    folds_path: str
    columns: List[Dict[str, Any]]
    leakage_policy: str   # exclude (default) | keep
    leakage_screen: List[Dict[str, Any]]
    # a hash of the data file, the schema and the planning prompts: identical
    # inputs, identical fingerprint, so a cached plan can be reused
    plan_fingerprint: str
    reuse_plan: bool
    # review: pause after the config is drafted until a person approves the plan.
    # approved: the plan has been reviewed; resplit: the reviewer changed the split
    review: bool
    approved: bool
    resplit: bool
    messages: Annotated[Sequence[BaseMessage], add_messages]


def target_column(state: InformationState) -> str:
    return [col for col in state["schema"].columns if col.is_target][0].name


# what the schema says the target is, in the profiler's terms: it settles
# regression versus classification for an integer target with a few levels,
# which the values alone cannot
_DECLARED_TARGET_TYPE = {
    DataType.NUMERIC: "numeric",
    DataType.ORDINAL: "numeric",
    DataType.CATEGORICAL: "categorical",
}


def profile(data_path: str, schema: DataSchema) -> Tuple[str, dict]:
    target = next(col for col in schema.columns if col.is_target)
    result = profile_dataset(
        pd.read_csv(data_path),
        target=target.name,
        # the descriptions the user wrote go into the COLUMNS table, so the
        # planners read "number of bedrooms" next to the numbers
        schema={col.name: col.description for col in schema.columns if col.description},
        target_type=_DECLARED_TARGET_TYPE.get(target.data_type),
    )
    # the whole profile, not just the notes: the preprocessing requirements are
    # derived from per-column statistics the notes do not carry
    return render_profile(result), result


def parse_json(text: str) -> dict:
    text = text.replace("```json", "").replace("```", "")
    payload, _ = json.JSONDecoder().raw_decode(text[text.find("{"):])
    return payload


PLANS_DIR = "_plans"


def _fingerprint(data_path: str, schema: DataSchema) -> str:
    digest = hashlib.sha256()
    with open(data_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    for text in (schema.model_dump_json(), split_planner_prompt, config_generator_prompt):
        digest.update(text.encode("utf-8"))
    return digest.hexdigest()[:16]


def _plan_cache(state) -> Path:
    return Path(state["run_dir"]).parent / PLANS_DIR / f"{state['plan_fingerprint']}.json"


def _reuse_plan(state) -> bool:
    return bool(state.get("reuse_plan")) or os.environ.get("AUTOML_REUSE_PLAN") == "1"


def _cached(state, key: str):
    path = _plan_cache(state)
    if not _reuse_plan(state) or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get(key)
    except (ValueError, OSError):
        return None


def _remember(state, key: str, value: dict) -> None:
    path = _plan_cache(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    stored = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    stored[key] = value
    path.write_text(json.dumps(stored, indent=2), encoding="utf-8")


def place_run(state: InformationState) -> InformationState:
    """Give the run its id and directory before anything writes a file.

    Assigned in code: the config generator used to supply the id and answered 1
    every time, so every run wrote into runs/1. Placing the run first also lets
    the split live inside it rather than next to the data, where two datasets in
    one folder overwrote each other's train.csv.
    """
    run_id = state.get("run_id") or int(datetime.now().strftime("%Y%m%d%H%M%S"))
    run_dir = Path(state.get("run_dir") or Path("runs") / str(run_id))
    run_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = _fingerprint(state["data_path"], state["schema"])
    print(f"run {run_id} -> {run_dir} (plan fingerprint {fingerprint})")
    return {"run_id": run_id, "run_dir": str(run_dir), "plan_fingerprint": fingerprint}


def summarize(state: InformationState) -> InformationState:
    # whole frame: the planner needs to see the ordering and the entities before the cut
    summary, _ = profile(state["data_path"], state["schema"])
    return {"full_summary": summary}


def plan_split(state: InformationState) -> InformationState:
    cached = _cached(state, "split_plan")
    if cached is not None:
        split_plan = SplitPlan(**cached)
        print(f"split plan reused from {_plan_cache(state)}")
        return {"split_plan": split_plan}

    model = hooks.llm_for(state["run_id"], "planner")
    response = model.invoke(
        [
            SystemMessage(content=split_planner_prompt),
            HumanMessage(content=state["full_summary"]),
        ]
    )

    split_plan = SplitPlan(**parse_json(message_text(response)))
    _remember(state, "split_plan", split_plan.model_dump())

    print(split_plan)
    return {"split_plan": split_plan}


def apply_split(state: InformationState) -> InformationState:
    data_frame = pd.read_csv(state["data_path"])
    train, test = apply_split_plan(data_frame, state["split_plan"])
    train_path, test_path = write_split(train, test, Path(state["run_dir"]) / "splits")

    print(f"split: {len(train)} train -> {train_path} | {len(test)} test -> {test_path}")
    return {"train_path": train_path, "test_path": test_path}


def summarize_train(state: InformationState) -> InformationState:
    # everything downstream profiles the training set only, so the test set
    # never influences a modelling decision
    summary, full = profile(state["train_path"], state["schema"])
    notes = full["modelling_notes"]

    # decided once, from the numbers, so the models that depend on them do not
    # re-derive them differently on every run
    requirements = derive_requirements(full)
    for requirement in requirements:
        print(f"requirement: {requirement.issue}")

    # what every candidate is told about the columns it will receive; the
    # profile's roles say more than a dtype does
    columns = [
        {
            "name": name,
            "kind": column["role"],
            "dtype": column.get("dtype", ""),
            "n_unique": column.get("n_unique"),
            "missing_pct": column.get("missing_pct"),
            "semantic_type": column.get("semantic_type"),
        }
        for name, column in full["columns"].items()
    ]

    # the profile's recommendations, and every column the schema marks as an
    # identifier or as ignored
    drop_columns = list(notes.get("recommend_drop", []))
    for column in state["schema"].columns:
        if column.role in ("identifier", "ignore") and column.name not in drop_columns:
            drop_columns.append(column.name)

    return {
        "summary": summary,
        "target": target_column(state),
        "task_type": notes["task_type"],
        "drop_columns": drop_columns,
        "preprocessing_requirements": requirements,
        "columns": columns,
    }


def screen_leakage(state: InformationState) -> InformationState:
    """Look for the target inside the features before any model is fitted.

    A formula of one or two columns that reproduces the target on held-out rows
    is a leak the data shows on its face, and catching it here means no model
    ever trains on it. What happens to the columns is decided in prepare_run,
    where the run exists to record it; this node only measures, and tells the
    planners what it found by adding it to the profile they read.
    """
    frame = pd.read_csv(state["train_path"], low_memory=False)
    hits = relation_screen(
        frame, state["target"], state["task_type"], state["columns"], seed=state["split_plan"].random_seed
    )
    declared = {
        column.name for column in state["schema"].columns if column.available_at_prediction
    }
    policy = state.get("leakage_policy") or "exclude"
    for hit in hits:
        if all(column in declared for column in hit["columns"]):
            hit["outcome"] = "kept: the schema declares these known at prediction time"
        elif policy == "exclude":
            hit["outcome"] = "excluded from the run: not declared known at prediction time"
        else:
            hit["outcome"] = "kept by the run's leakage policy, though it reconstructs the target"
        print(f"leakage screen: {', '.join(hit['columns'])}: {hit['formula']} "
              f"({hit['measure']} {hit['score']:.4f}) -> {hit['outcome']}")

    return {"leakage_screen": hits, "summary": state["summary"] + render_screen(hits)}


def _sandbox(state):
    """The run's sandbox host, or None when candidates run on this machine."""
    return hooks.sandbox_for(state["run_id"]) if state.get("backend") == "sandbox" else None


def probe_env(state: InformationState) -> InformationState:
    backend = state.get("backend", "subprocess")
    environment = probe_environment(backend, _sandbox(state))
    print(f"environment: {', '.join(f'{k} {v}' for k, v in sorted(environment.items()))}")
    return {"environment": environment}


def generate_config(state: InformationState) -> InformationState:
    if state.get("approved"):
        # back here after a reviewer changed the split: keep the config they approved
        return {}
    cached = _cached(state, "config")
    if cached is not None:
        config = Configs(**cached)
        print(f"config reused from {_plan_cache(state)}")
        return {"config": config}

    model = hooks.llm_for(state["run_id"], "planner")
    usable = available_model_names(state["environment"])
    # stating this up front beats pruning afterwards: a pruned list loses the
    # model the generator ranked first and leaves a short, unbalanced slate
    prompt = (
        state["summary"]
        + "\n\n## AVAILABLE IN THIS ENVIRONMENT\n"
        + f"Choose only from: {', '.join(usable)}"
    )
    response = model.invoke(
        [
            SystemMessage(content=config_generator_prompt),
            HumanMessage(content=prompt),
        ]
    )

    config = Configs(**parse_json(message_text(response)))
    _remember(state, "config", config.model_dump())

    print(config)
    return {"config": config}


def review_plan(state: InformationState) -> InformationState:
    """Pause for a person to review the drafted plan, when the caller asked for it.

    interrupt() stops the graph here and the checkpointer keeps it; nothing runs
    and nothing is held in memory while it waits. It resumes with the reviewer's
    edits, {"split_plan": {...}, "config": {...}}, either optional and either
    partial: each is merged over the drafted values and checked by building the
    model. A changed split sends the graph back to apply_split; the planners are
    not asked again.
    """
    if not state.get("review") or state.get("approved"):
        return {"resplit": False}

    edits = interrupt({
        # task_type and target are read-only: they come from the locked schema and
        # the profile, and changing either means a different job (docs/PHASE5.md §8.5)
        "task_type": state["task_type"],
        "target": state["target"],
        "split_plan": state["split_plan"].model_dump(),
        "config": state["config"].model_dump(),
        "leakage_screen": state.get("leakage_screen") or [],
        "choices": review_choices(state),
    }) or {}
    split_plan = SplitPlan(**{**state["split_plan"].model_dump(), **(edits.get("split_plan") or {})})
    config = Configs(**{**state["config"].model_dump(), **(edits.get("config") or {})})
    return {
        "split_plan": split_plan,
        "config": config,
        "approved": True,
        "resplit": split_plan != state["split_plan"],
    }


def review_choices(state: InformationState) -> dict:
    """What the reviewer may pick from, straight out of the registries that enforce it.

    The console offers these and nothing else, so an edit cannot name a split
    method apply_split_plan would reject, a metric the scorer has never heard of,
    or a model family this environment cannot import. Unavailable models are
    listed with the reason rather than hidden: dropping them silently is how a
    run ends up training four families when you asked for five.
    """
    environment = state.get("environment") or {}
    _, unavailable = resolve_models(ALL_MODELS, environment)
    regression = state["task_type"] == "regression"
    return {
        "models": [{"name": model, "available": model not in unavailable, "why": unavailable.get(model)}
                   for model in ALL_MODELS],
        "metrics": [metric for metric in METRIC_DIRECTION
                    if (metric in REGRESSION_METRICS) == regression],
        # which way is better, so the console can rank without a table of its own
        "directions": {metric: direction for metric, direction in METRIC_DIRECTION.items()
                       if (metric in REGRESSION_METRICS) == regression},
        "split_methods": list(SPLIT_METHODS),
    }


def route_review(state: InformationState) -> str:
    return "apply_split" if state.get("resplit") else "resolve_environment"


def resolve_environment(state: InformationState) -> InformationState:
    """Pin the libraries the run will use, and drop models nothing can import.

    Done once, here, before any worker starts: installing a package while the
    models are training would mean the attempts either side of it ran under
    different conditions, and five parallel workers installing into one
    environment is a race.
    """
    backend = state.get("backend", "subprocess")
    # install what the chosen models need, once, before anything trains. A
    # sandbox is the exception: it has no network by design, so a missing
    # package there means rebuilding the image, not installing at runtime
    policy = state.get("provision_policy") or (
        PROVISION_NEVER if backend == "sandbox" else PROVISION_ONCE
    )
    config = state["config"]
    environment = state["environment"]

    installed, notes = provision(config.models, environment, policy, backend)
    if installed:
        environment = probe_environment(backend, _sandbox(state))

    available, unavailable = resolve_models(config.models, environment)
    if unavailable:
        notes.append(
            "dropped before the run: "
            + ", ".join(f"{model} ({reason})" for model, reason in sorted(unavailable.items()))
        )
    if len(available) < 2:
        notes.append(f"only {len(available)} model(s) can run here, so there is little to compare")

    for note in notes:
        print(f"environment: {note}")

    return {
        "environment": environment,
        "unavailable_models": unavailable,
        "environment_notes": notes,
        "config": config.model_copy(update={"models": available}),
    }


def prepare_run(state: InformationState) -> InformationState:
    """Freeze the run's folds and apply the leakage screen, before anything is scored.

    The folds are computed once and saved, and the baseline and every candidate
    read the same file: rebuilding a splitter per model, even from one seed,
    left the comparability of their scores to chance.
    """
    run_id, run_dir = state["run_id"], Path(state["run_dir"])

    plan = state["split_plan"]
    train = pd.read_csv(state["train_path"], low_memory=False)
    groups = train[plan.group_column] if plan.cv_strategy == "group_kfold" and plan.group_column else None
    folds = build_folds(plan.cv_strategy, plan.cv_folds, plan.random_seed, train[state["target"]], groups)
    folds_path = save_folds(run_dir / "folds.npz", folds)
    print(f"run {run_id}: {len(folds)} {plan.cv_strategy} folds frozen -> {folds_path}")

    # the screen's verdicts become run-wide exclusions, the same record a leak
    # confirmed later by a model's leak check is written to
    for hit in state.get("leakage_screen") or []:
        if hit.get("outcome", "").startswith("excluded"):
            hooks.emit(run_id, "exclusion", columns=hit["columns"], found_by="leakage screen", reason=hit["formula"])
            add_exclusions(run_dir, hit["columns"], {
                "reason": (
                    f"leakage screen before any model: {hit['formula']} reproduces the target "
                    f"({hit['measure']} {hit['score']:.4f} on held-out rows)"
                ),
                "found_by": "leakage screen",
                "measured_with": "relation screen",
                "score": hit["score"],
                "measure": hit["measure"],
            })

    return {"run_id": run_id, "run_dir": str(run_dir), "folds_path": str(folds_path)}


def compute_baseline(state: InformationState) -> InformationState:
    # once, centrally: every worker is measured against the same floor, on the
    # same frozen folds
    baseline = run_baseline(
        pd.read_csv(state["train_path"], low_memory=False),
        state["target"],
        state["task_type"],
        state["split_plan"],
        state["config"],
        folds=load_folds(state["folds_path"]),
    )

    print(f"baseline: {baseline.strategy} -> {baseline.cv_scores}")
    return {"baseline": baseline}


def build_context(state: InformationState) -> InformationState:
    config = state["config"]
    primary_metric = config.eval_matrics[0]
    target_values = pd.read_csv(state["train_path"], usecols=[state["target"]])[state["target"]]
    n_rows = len(target_values)

    # the target's own scale, so a guard can tell whether an attempt's
    # predictions and scores came back in the units they claim
    target_stats = {"n_unique": float(target_values.nunique())}
    if state["task_type"] == "regression":
        target_stats.update(
            {
                "min": float(target_values.min()),
                "max": float(target_values.max()),
                "mean": float(target_values.mean()),
                "median": float(target_values.median()),
                "std": float(target_values.std()),
            }
        )

    resources = plan_resources(state["train_path"], len(config.models), _sandbox(state))
    concurrency = int(os.environ.get("AUTOML_CONCURRENCY") or resources.max_concurrency)
    print(f"resources: {resources.reason}" + (f"; concurrency overridden to {concurrency}" if concurrency != resources.max_concurrency else ""))

    context = RunContext(
        run_id=state["run_id"],
        run_dir=state["run_dir"],
        train_path=state["train_path"],
        test_path=state["test_path"],
        split_plan=state["split_plan"],
        folds_path=state["folds_path"],
        columns=state["columns"],
        available_at_prediction=[
            column.name for column in state["schema"].columns
            if column.available_at_prediction and not column.is_target
        ],
        leakage_policy=state.get("leakage_policy") or "exclude",
        leakage_screen=state.get("leakage_screen") or [],
        summary=state["summary"],
        target=state["target"],
        task_type=state["task_type"],
        target_stats=target_stats,
        drop_columns=state["drop_columns"],
        preprocessing_requirements=state["preprocessing_requirements"],
        baseline=state["baseline"],
        eval_matrics=config.eval_matrics,
        primary_metric=primary_metric,
        metric_direction=METRIC_DIRECTION[primary_metric],
        improvement_metric=config.improvement_metric,
        improvement_delta=config.improvement_delta,
        improvement_mode=config.improvement_mode,
        early_stopping_patience=config.early_stopping_patience,
        max_tries=config.max_tries,
        # a booster on 200k rows needs longer than a ridge on 800, and parallel
        # workers share one machine, so neither of these can be a constant
        time_budget_seconds=600 if n_rows > 100_000 else 300,
        n_jobs=resources.n_jobs,
        max_concurrency=concurrency,
        worker_memory_mb=resources.worker_memory_mb,
        reserve_memory_mb=resources.reserve_memory_mb,
        reserve_cpus=resources.reserve_cpus,
        resource_plan=resources.reason,
        plan_fingerprint=state.get("plan_fingerprint", ""),
        backend=state.get("backend", "subprocess"),
        environment=state["environment"],
        unavailable_models=state["unavailable_models"],
        environment_notes=state["environment_notes"],
    )

    # written by the graph itself, so a crashed run can be resumed by any caller
    run_dir = Path(context.run_dir)
    (run_dir / "context.json").write_text(context.model_dump_json(indent=2), encoding="utf-8")
    (run_dir / "config.json").write_text(config.model_dump_json(indent=2), encoding="utf-8")

    print(f"context: run {context.run_id} -> {context.run_dir}")
    return {"context": context}


graph = StateGraph(InformationState)
graph.add_node("place_run", place_run)
graph.add_node("summarize", summarize)
graph.add_node("plan_split", plan_split)
graph.add_node("apply_split", apply_split)
graph.add_node("summarize_train", summarize_train)
graph.add_node("screen_leakage", screen_leakage)
graph.add_node("probe_env", probe_env)
graph.add_node("generate_config", generate_config)
graph.add_node("review_plan", review_plan)
graph.add_node("resolve_environment", resolve_environment)
graph.add_node("prepare_run", prepare_run)
graph.add_node("compute_baseline", compute_baseline)
graph.add_node("build_context", build_context)
graph.add_edge(START, "place_run")
graph.add_edge("place_run", "summarize")
graph.add_edge("summarize", "plan_split")
graph.add_edge("plan_split", "apply_split")
graph.add_edge("apply_split", "summarize_train")
graph.add_edge("summarize_train", "screen_leakage")
graph.add_edge("screen_leakage", "probe_env")
graph.add_edge("probe_env", "generate_config")
graph.add_edge("generate_config", "review_plan")
graph.add_conditional_edges("review_plan", route_review, ["apply_split", "resolve_environment"])
graph.add_edge("resolve_environment", "prepare_run")
graph.add_edge("prepare_run", "compute_baseline")
graph.add_edge("compute_baseline", "build_context")
graph.add_edge("build_context", END)

app = graph.compile()


if __name__ == "__main__":
    from test_values import data_schema

    result = app.invoke({"data_path": "data/train.csv", "schema": data_schema})
    print(result["context"].model_dump_json(indent=2, exclude={"summary"}))
