import sys
from pathlib import Path

# Allow imports from project root when running this file directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import TypedDict, Annotated, Dict, List, Sequence, Tuple
import json

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_google_genai import ChatGoogleGenerativeAI
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
from utils.reusable.summary import profile_dataset, render_profile
from utils.reusable.splitting import apply_split_plan, write_split
from utils.reusable.baseline import run_baseline
from utils.reusable.requirements import derive_requirements
from utils.reusable.llm import message_text
from utils.reusable.metrics import METRIC_DIRECTION
from utils.reusable.environment import (
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


def summarize(state: InformationState) -> InformationState:
    # whole frame: the planner needs to see the ordering and the entities before the cut
    summary, _ = profile(state["data_path"], state["schema"])
    return {"full_summary": summary}


def plan_split(state: InformationState) -> InformationState:
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
    response = model.invoke(
        [
            SystemMessage(content=split_planner_prompt),
            HumanMessage(content=state["full_summary"]),
        ]
    )

    split_plan = SplitPlan(**parse_json(message_text(response)))

    print(split_plan)
    return {"split_plan": split_plan}


def apply_split(state: InformationState) -> InformationState:
    data_frame = pd.read_csv(state["data_path"])
    train, test = apply_split_plan(data_frame, state["split_plan"])
    train_path, test_path = write_split(train, test, Path(state["data_path"]).parent / "splits")

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

    return {
        "summary": summary,
        "target": target_column(state),
        "task_type": notes["task_type"],
        "drop_columns": notes.get("recommend_drop", []),
        "preprocessing_requirements": requirements,
    }


def probe_env(state: InformationState) -> InformationState:
    backend = state.get("backend", "subprocess")
    environment = probe_environment(backend)
    print(f"environment: {', '.join(f'{k} {v}' for k, v in sorted(environment.items()))}")
    return {"environment": environment}


def generate_config(state: InformationState) -> InformationState:
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
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

    print(config)
    return {"config": config}


def resolve_environment(state: InformationState) -> InformationState:
    """Pin the libraries the run will use, and drop models nothing can import.

    Done once, here, before any worker starts: installing a package while the
    models are training would mean the attempts either side of it ran under
    different conditions, and five parallel workers installing into one
    environment is a race.
    """
    backend = state.get("backend", "subprocess")
    # install what the chosen models need, once, before anything trains. Docker
    # is the exception: its training container has no network by design, so a
    # missing package there means rebuilding the image, not installing at runtime
    policy = state.get("provision_policy") or (
        PROVISION_NEVER if backend == "docker" else PROVISION_ONCE
    )
    config = state["config"]
    environment = state["environment"]

    installed, notes = provision(config.models, environment, policy, backend)
    if installed:
        environment = probe_environment(backend)

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


def compute_baseline(state: InformationState) -> InformationState:
    # once, centrally: every worker is measured against the same floor
    baseline = run_baseline(
        pd.read_csv(state["train_path"]),
        state["target"],
        state["task_type"],
        state["split_plan"],
        state["config"],
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

    context = RunContext(
        run_id=config.id,
        run_dir=str(Path("runs") / str(config.id)),
        train_path=state["train_path"],
        test_path=state["test_path"],
        split_plan=state["split_plan"],
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
        n_jobs=2,
        backend=state.get("backend", "subprocess"),
        environment=state["environment"],
        unavailable_models=state["unavailable_models"],
        environment_notes=state["environment_notes"],
    )

    print(f"context: run {context.run_id} -> {context.run_dir}")
    return {"context": context}


graph = StateGraph(InformationState)
graph.add_node("summarize", summarize)
graph.add_node("plan_split", plan_split)
graph.add_node("apply_split", apply_split)
graph.add_node("summarize_train", summarize_train)
graph.add_node("probe_env", probe_env)
graph.add_node("generate_config", generate_config)
graph.add_node("resolve_environment", resolve_environment)
graph.add_node("compute_baseline", compute_baseline)
graph.add_node("build_context", build_context)
graph.add_edge(START, "summarize")
graph.add_edge("summarize", "plan_split")
graph.add_edge("plan_split", "apply_split")
graph.add_edge("apply_split", "summarize_train")
graph.add_edge("summarize_train", "probe_env")
graph.add_edge("probe_env", "generate_config")
graph.add_edge("generate_config", "resolve_environment")
graph.add_edge("resolve_environment", "compute_baseline")
graph.add_edge("compute_baseline", "build_context")
graph.add_edge("build_context", END)

app = graph.compile()


if __name__ == "__main__":
    from test_values import data_schema

    result = app.invoke({"data_path": "data/train.csv", "schema": data_schema})
    print(result["context"].model_dump_json(indent=2, exclude={"summary"}))
