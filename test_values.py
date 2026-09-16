import sys
from pathlib import Path

# Allow imports from project root when running this file directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import DataSchema, DataColumn, DataType, Configs, RunContext

FIXTURES = Path(__file__).resolve().parent / "fixtures"


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

# One real information-agent run, frozen so the code generation stages can be
# built and replayed without spending two LLM calls every time. Regenerate both
# files by running the information graph and dumping result["context"] and
# result["config"].
context = RunContext.model_validate_json((FIXTURES / "run_context.json").read_text(encoding="utf-8"))
config = Configs.model_validate_json((FIXTURES / "config.json").read_text(encoding="utf-8"))

# the training-set profile the workers are given; same object the context carries
summary = context.summary
