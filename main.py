from typing import TypedDict, Annotated, List, Sequence

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_core.messages import BaseMessage

from models import Configs, DataSchema, DataColumn, DataType, SplitPlan, BaselineResult, RunContext
from information_agent.base import app as information_app
from code_gen_eval.base import app as code_gen_eval_app


class AutoMLState(TypedDict):
    topic: str
    # optional: place the run; it is resumable from this directory with
    # code_gen_eval.resume.resume_run(run_dir)
    run_id: int
    run_dir: str
    # pause after planning until a person approves the plan (needs a checkpointer)
    review: bool
    # subprocess (this machine) or sandbox (hooks.sandbox_for)
    backend: str
    data_path: str
    schema: DataSchema
    full_summary: str
    summary: str
    train_path: str
    test_path: str
    target: str
    task_type: str
    drop_columns: List[str]
    config: Configs
    split_plan: SplitPlan
    baseline: BaselineResult
    context: RunContext
    report: str
    messages: Annotated[Sequence[BaseMessage], add_messages]


graph = StateGraph(AutoMLState)

graph.add_node("information", information_app)
graph.add_node("code_gen_eval", code_gen_eval_app)

graph.add_edge(START, "information")
graph.add_edge("information", "code_gen_eval")
graph.add_edge("code_gen_eval", END)

app = graph.compile()


def checkpoint_types() -> list:
    """Every class defined in models/: the checkpointer's allowlist.

    LangGraph is moving to refuse loading any class from a checkpoint that is
    not listed, so a caller that compiles `graph` with a checkpointer passes
    JsonPlusSerializer(allowed_msgpack_modules=checkpoint_types()).
    """
    import importlib
    import inspect
    import pkgutil

    import models

    found = []
    for info in pkgutil.iter_modules(models.__path__):
        module = importlib.import_module(f"models.{info.name}")
        found += [cls for _, cls in inspect.getmembers(module, inspect.isclass) if cls.__module__ == module.__name__]
    return found


if __name__ == "__main__":
    data_columns = [
        DataColumn(name="POSTED_BY", data_type=DataType.CATEGORICAL, description="Who posted the property listing (Owner, Dealer, or Builder)"),
        DataColumn(name="UNDER_CONSTRUCTION", data_type=DataType.CATEGORICAL, description="Whether the property is under construction (0 = no, 1 = yes)"),
        DataColumn(name="RERA", data_type=DataType.CATEGORICAL, description="Whether the property is RERA registered (0 = no, 1 = yes)"),
        DataColumn(name="BHK_NO.", data_type=DataType.NUMERIC, description="Number of bedrooms in the property"),
        DataColumn(name="BHK_OR_RK", data_type=DataType.CATEGORICAL, description="Property layout type (BHK or RK)"),
        DataColumn(name="SQUARE_FT", data_type=DataType.NUMERIC, description="Property area in square feet"),
        DataColumn(name="READY_TO_MOVE", data_type=DataType.CATEGORICAL, description="Whether the property is ready to move in (0 = no, 1 = yes)"),
        DataColumn(name="RESALE", data_type=DataType.CATEGORICAL, description="Whether the property is a resale listing (0 = no, 1 = yes)"),
        DataColumn(name="ADDRESS", data_type=DataType.TEXT, description="Property location address (locality and city)"),
        DataColumn(name="LONGITUDE", data_type=DataType.LONG, description="Geographic longitude of the property"),
        DataColumn(name="LATITUDE", data_type=DataType.LATITUDE, description="Geographic latitude of the property"),
        DataColumn(name="TARGET(PRICE_IN_LACS)", data_type=DataType.NUMERIC, description="Property price in lakh rupees", is_target=True),
    ]
    data_schema = DataSchema(name="housing_data", columns=data_columns, description="Indian housing price prediction dataset")

    result = app.invoke(
        {
            "topic": "House prices prediction",
            "data_path": "data/train.csv",
            "schema": data_schema,
        }
    )
    print("===============> Report <===============")
    print(result["report"])
