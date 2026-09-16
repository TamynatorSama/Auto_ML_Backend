from pydantic import BaseModel
from enum import Enum
from typing import List



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

class DataSchema(BaseModel):
    name: str
    columns: List[DataColumn]
    description: str
    need_real_world_info: bool = False
