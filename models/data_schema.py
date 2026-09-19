from pydantic import BaseModel, model_validator
from enum import Enum
from typing import List, Literal, Optional



class DataType(Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEXT = "text"
    JSON = "json"
    ORDINAL = "ordinal"
    LONG = "long"
    LATITUDE = "latitude"

class DataColumn(BaseModel):
    name: str
    data_type: DataType
    description: str
    # identifier and ignore columns are dropped before any model sees them
    role: Literal["feature", "identifier", "ignore", "target"] = "feature"
    # kept in step with role, so schemas written with is_target keep working
    is_target: bool = False
    # whether this value is known at the moment a prediction is made. Only a
    # person can say: `units` is a leak when predicting revenue after the fact
    # and a legitimate input when pricing an order. None means not declared,
    # and a column the run finds reconstructing the target is then handled by
    # the run's leakage policy.
    available_at_prediction: Optional[bool] = None

    @model_validator(mode="after")
    def _target_role(self):
        if self.is_target:
            self.role = "target"
        elif self.role == "target":
            self.is_target = True
        return self

class DataSchema(BaseModel):
    name: str
    columns: List[DataColumn]
    description: str
    need_real_world_info: bool = False
