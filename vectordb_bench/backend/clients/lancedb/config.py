from enum import Enum
from typing import Optional
from pydantic import SecretStr, BaseModel
from ..api import DBConfig, DBCaseConfig, IndexType

class MetricType(str, Enum):
    L2 = "l2"
    COSINE = "cosine"
    DOT = "dot"

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

class LanceDBIndexType(str, Enum):
    NONE = "NONE"
    AUTO = IndexType.AUTOINDEX.value
    IVF_PQ = IndexType.IVFPQ.value
    HNSW = IndexType.HNSW.value


class LanceDBIndexConfig(BaseModel, DBCaseConfig):
    index: LanceDBIndexType = LanceDBIndexType.NONE
    metric_type: MetricType = MetricType.L2
    num_partitions: int = 0  # Default (when 0): sqrt(num_rows)
    num_sub_vectors: int = 0  # Default (when 0): dim/16 or dim/8
    nbits: int = 8  # Must be 4 or 8
    sample_rate: int = 256
    max_iterations: int = 50

    def index_param(self) -> dict:
        # See https://lancedb.github.io/lancedb/python/python/#lancedb.table.Table.create_index
        params = {
            "metric": self.metric_type.value,
            "num_bits": self.nbits,
            "sample_rate": self.sample_rate,
            "max_iterations": self.max_iterations
        }

        if self.num_partitions > 0:
            params["num_partitions"] = self.num_partitions
        if self.num_sub_vectors > 0:
            params["num_sub_vectors"] = self.num_sub_vectors

        return params

    def search_param(self) -> dict:
        return {"metric_type": self.metric_type.value}

class LanceDBNoIndexConfig(LanceDBIndexConfig):
    index: LanceDBIndexType = LanceDBIndexType.NONE
    def index_param(self) -> dict:
        return {}

class LanceDBAutoIndexConfig(LanceDBIndexConfig):
    index: LanceDBIndexType = LanceDBIndexType.AUTO
    def index_param(self) -> dict:
        return {}

class LanceDBIVFIndexConfig(LanceDBIndexConfig):
    index: LanceDBIndexType = LanceDBIndexType.IVF_PQ

class LanceDBHNSWIndexConfig(LanceDBIndexConfig):
    index: LanceDBIndexType = LanceDBIndexType.HNSW
    m: int = 0
    ef_construction: int = 0

    def index_param(self) -> dict:
        params = LanceDBIndexConfig.index_param(self)

        # See https://lancedb.github.io/lancedb/python/python/#lancedb.index.HnswSq
        params["index_type"] = "IVF_HNSW_SQ"

        if self.m > 0:
            params["m"] = self.m
        if self.ef_construction > 0:
            params["ef_construction"] = self.ef_construction

        return params

_lancedb_case_config = {
    IndexType.AUTOINDEX: LanceDBAutoIndexConfig,
    IndexType.IVFPQ: LanceDBIVFIndexConfig,
    IndexType.HNSW: LanceDBHNSWIndexConfig,
    "NONE": LanceDBNoIndexConfig,
}
