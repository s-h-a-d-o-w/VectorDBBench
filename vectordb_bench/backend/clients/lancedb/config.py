from enum import Enum
from typing import Optional
from pydantic import SecretStr
from ..api import DBConfig, DBCaseConfig

# class MetricType(str, Enum):
#     L2 = "L2"
#     COSINE = "COSINE"
#     DOT = "DOT"

class LanceDBConfig(DBConfig):
    """LanceDB connection configuration."""
    db_label: str
    uri: str
    token: Optional[SecretStr] = None

    def to_dict(self) -> dict:
        return {
            "uri": self.uri,
            "token": self.token.get_secret_value() if self.token else None,
        }

# class LanceDBIndexConfig(DBCaseConfig):
#     """LanceDB index configuration."""
#     metric_type: MetricType = MetricType.COSINE
#     create_index: bool = True
