from pydantic import BaseModel
from enum import Enum
from typing import List, Optional



class DataType(Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEXT = "text"
    JSON = "json"
    ORDINAL = "ordinal"
    LONG = "long"
    LATITUDE = "latitude"
    IS_TARGET = "is_target"

class DataColumn(BaseModel):
    name: str
    data_type: DataType
    description: str
    is_target: bool = False
    # whether this value is known at the moment a prediction is made. Only a
    # person can say: `units` is a leak when predicting revenue after the fact
    # and a legitimate input when pricing an order. None means not declared,
    # and a column the run finds reconstructing the target is then handled by
    # the run's leakage policy.
    available_at_prediction: Optional[bool] = None

class DataSchema(BaseModel):
    name: str
    columns: List[DataColumn]
    description: str
    need_real_world_info: bool = False
