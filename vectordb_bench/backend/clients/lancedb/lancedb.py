"""Wrapper around the LanceDB vector database over VectorDB"""

import logging
from contextlib import contextmanager
from typing import List, Optional, Tuple

import lancedb
from lancedb.pydantic import LanceModel
import pyarrow as pa

from ..api import VectorDB
from .config import LanceDBConfig, LanceDBIndexConfig, LanceDBIndexType

log = logging.getLogger(__name__)

class Vector(LanceModel):
    id: int
    vector: List[float]

class LanceDB(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: LanceDBConfig,
        db_case_config: LanceDBIndexConfig,
        collection_name: str = "vector_bench_test",
        drop_old: bool = False,
        **kwargs,
    ):
        """Initialize wrapper around the LanceDB vector database."""
        self.name = "LanceDB"
        self.db_config = db_config
        self.case_config = db_case_config
        self.table_name = collection_name
        self.dim = dim
        self.uri = db_config['uri']  # Store the URI instead of connection

        db = lancedb.connect(self.uri)

        if drop_old:
            try:
                db.drop_table(self.table_name)
            except Exception as e:
                log.warning(f"Failed to drop table {self.table_name}: {e}")

        try:
            db.open_table(self.table_name)
        except Exception as e:
            schema = pa.schema([
                pa.field("id",     pa.int64()),
                pa.field("vector", pa.list_(pa.float64(), list_size=self.dim))
            ])
            db.create_table(
                self.table_name,
                schema=schema,
                mode="overwrite"
            )

    @contextmanager
    def init(self):
        self.db = lancedb.connect(self.uri)
        self.table = self.db.open_table(self.table_name)
        yield
        self.db = None
        self.table = None

    def insert_embeddings(
        self,
        embeddings: List[List[float]],
        metadata: List[int],
    ) -> Tuple[int, Optional[Exception]]:
        """Insert embeddings with their associated metadata."""
        try:
            data = [
                {"id": meta, "vector": emb}
                for meta, emb in zip(metadata, embeddings)
            ]
            self.table.add(data)
            return len(metadata), None
        except Exception as e:
            log.warning(f"Failed to insert data into LanceDB table ({self.table_name}), error: {e}")
            return 0, e

    def search_embedding(
        self,
        query: List[float],
        k: int = 100,
        filters: Optional[dict] = None,
    ) -> List[int]:
        if filters:
            results = self.table.search(query).where(f"id >= {filters['id']}", prefilter=True).limit(k).to_list()
        else:
            results = self.table.search(query).limit(k).to_list()
        return [int(result["id"]) for result in results]

    def optimize(self, data_size: int | None = None):
        if self.table and hasattr(self, 'case_config'):
            if self.case_config.index != LanceDBIndexType.NONE:
                log.info(f"Creating index for LanceDB table ({self.table_name})")
                self.table.create_index(**self.case_config.index_param())
                # Better recall with IVF_PQ (though still bad).
                # But breaks HNSW: https://github.com/lancedb/lancedb/issues/2369
                self.table.optimize()
