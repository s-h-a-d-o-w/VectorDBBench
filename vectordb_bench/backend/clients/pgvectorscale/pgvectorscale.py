"""Wrapper around the Pgvectorscale vector database over VectorDB"""

import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg import Connection, Cursor, sql

from ..api import VectorDB
from .config import PgVectorScaleConfigDict, PgVectorScaleIndexConfig

log = logging.getLogger(__name__)


class PgVectorScale(VectorDB):
    """Use psycopg instructions"""

    conn: psycopg.Connection[Any] | None = None
    coursor: psycopg.Cursor[Any] | None = None

    _unfiltered_search: sql.Composed
    _filtered_search: sql.Composed

    def __init__(
        self,
        dim: int,
        db_config: PgVectorScaleConfigDict,
        db_case_config: PgVectorScaleIndexConfig,
        collection_name: str = "pg_vectorscale_collection",
        drop_old: bool = False,
        **kwargs,
    ):
        self.name = "PgVectorScale"
        self.db_config = db_config
        self.case_config = db_case_config
        self.table_name = collection_name
        self.dim = dim

        self._index_name = "pgvectorscale_index"
        self._primary_field = "id"
        self._vector_field = "embedding"

        self.conn, self.cursor = self._create_connection(**self.db_config)

        log.info(f"{self.name} config values: {self.db_config}\n{self.case_config}")
        if not any(
            (
                self.case_config.create_index_before_load,
                self.case_config.create_index_after_load,
            ),
        ):
            msg = (
                f"{self.name} config must create an index using create_index_before_load or create_index_after_load"
                f"{self.name} config values: {self.db_config}\n{self.case_config}"
            )
            log.error(msg)
            raise RuntimeError(msg)

        if drop_old:
            self._drop_index()
            self._drop_table()
            self._create_table(dim)
            if self.case_config.create_index_before_load:
                self._create_index()

        self.cursor.close()
        self.conn.close()
        self.cursor = None
        self.conn = None

    @staticmethod
    def _create_connection(**kwargs) -> tuple[Connection, Cursor]:
        conn = psycopg.connect(**kwargs)
        conn.cursor().execute("CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE")
        conn.commit()
        register_vector(conn)
        conn.autocommit = False
        cursor = conn.cursor()

        assert conn is not None, "Connection is not initialized"
        assert cursor is not None, "Cursor is not initialized"

        return conn, cursor

    @contextmanager
    def init(self) -> Generator[None, None, None]:
        self.conn, self.cursor = self._create_connection(**self.db_config)

        # index configuration may have commands defined that we should set during each client session
        session_options: dict[str, Any] = self.case_config.session_param()

        if len(session_options) > 0:
            for setting_name, setting_val in session_options.items():
                command = sql.SQL("SET {setting_name} " + "= {setting_val};").format(
                    setting_name=sql.Identifier(setting_name),
                    setting_val=sql.Identifier(str(setting_val)),
                )
                log.debug(command.as_string(self.cursor))
                self.cursor.execute(command)
            self.conn.commit()

        self._filtered_search = sql.Composed(
            [
                sql.SQL("SELECT id FROM public.{} WHERE id >= %s ORDER BY embedding ").format(
                    sql.Identifier(self.table_name),
                ),
                sql.SQL(self.case_config.search_param()["metric_fun_op"]),
                sql.SQL(" %s::vector LIMIT %s::int"),
            ],
        )

        self._unfiltered_search = sql.Composed(
            [
                sql.SQL("SELECT id FROM public.{} ORDER BY embedding ").format(
                    sql.Identifier(self.table_name),
                ),
                sql.SQL(self.case_config.search_param()["metric_fun_op"]),
                sql.SQL(" %s::vector LIMIT %s::int"),
            ],
        )

        try:
            yield
        finally:
            self.cursor.close()
            self.conn.close()
            self.cursor = None
            self.conn = None

    def _drop_table(self):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"
        log.info(f"{self.name} client drop table : {self.table_name}")

        self.cursor.execute(
            sql.SQL("DROP TABLE IF EXISTS public.{table_name}").format(
                table_name=sql.Identifier(self.table_name),
            ),
        )
        self.conn.commit()

    def optimize(self, data_size: int | None = None):
        self._post_insert()

    def _post_insert(self):
        log.info(f"{self.name} post insert before optimize")
        if self.case_config.create_index_after_load:
            self._drop_index()
            self._create_index()

    def _drop_index(self):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"
        log.info(f"{self.name} client drop index : {self._index_name}")

        drop_index_sql = sql.SQL("DROP INDEX IF EXISTS {index_name}").format(
            index_name=sql.Identifier(self._index_name),
        )
        log.debug(drop_index_sql.as_string(self.cursor))
        self.cursor.execute(drop_index_sql)
        self.conn.commit()

    def _create_index(self):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"
        log.info(f"{self.name} client create index : {self._index_name}")

        index_param: dict[str, Any] = self.case_config.index_param()

        options = []
        for option_name, option_val in index_param["options"].items():
            if option_val is not None:
                options.append(
                    sql.SQL("{option_name} = {val}").format(
                        option_name=sql.Identifier(option_name),
                        val=sql.Identifier(str(option_val)),
                    ),
                )

        num_bits_per_dimension = "2" if self.dim < 900 else "1"
        options.append(
            sql.SQL("{option_name} = {val}").format(
                option_name=sql.Identifier("num_bits_per_dimension"),
                val=sql.Identifier(num_bits_per_dimension),
            ),
        )

        with_clause = sql.SQL("WITH ({});").format(sql.SQL(", ").join(options)) if any(options) else sql.Composed(())

        index_create_sql = sql.SQL(
            """
            CREATE INDEX IF NOT EXISTS {index_name} ON public.{table_name}
            USING {index_type} (embedding {embedding_metric})
            """,
        ).format(
            index_name=sql.Identifier(self._index_name),
            table_name=sql.Identifier(self.table_name),
            index_type=sql.Identifier(index_param["index_type"].lower()),
            embedding_metric=sql.Identifier(index_param["metric"]),
        )
        index_create_sql_with_with_clause = (index_create_sql + with_clause).join(" ")
        log.debug(index_create_sql_with_with_clause.as_string(self.cursor))
        self.cursor.execute(index_create_sql_with_with_clause)
        self.conn.commit()

    def _create_table(self, dim: int):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        try:
            log.info(f"{self.name} client create table : {self.table_name}")

            self.cursor.execute(
                sql.SQL(
                    "CREATE TABLE IF NOT EXISTS public.{table_name} (id BIGINT PRIMARY KEY, embedding vector({dim}));",
                ).format(table_name=sql.Identifier(self.table_name), dim=dim),
            )
            self.conn.commit()
        except Exception as e:
            log.warning(f"Failed to create pgvectorscale table: {self.table_name} error: {e}")
            raise e from None

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs: Any,
    ) -> tuple[int, Exception | None]:
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        try:
            metadata_arr = np.array(metadata)
            embeddings_arr = np.array(embeddings)

            with self.cursor.copy(
                sql.SQL("COPY public.{table_name} FROM STDIN (FORMAT BINARY)").format(
                    table_name=sql.Identifier(self.table_name),
                ),
            ) as copy:
                copy.set_types(["bigint", "vector"])
                for i, row in enumerate(metadata_arr):
                    copy.write_row((row, embeddings_arr[i]))
            self.conn.commit()

            if kwargs.get("last_batch"):
                self._post_insert()

            return len(metadata), None
        except Exception as e:
            log.warning(f"Failed to insert data into pgvector table ({self.table_name}), error: {e}")
            return 0, e

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
    ) -> list[int]:
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        q = np.asarray(query)
        if filters:
            gt = filters.get("id")
            result = self.cursor.execute(
                self._filtered_search,
                (gt, q, k),
                prepare=True,
                binary=True,
            )
        else:
            result = self.cursor.execute(self._unfiltered_search, (q, k), prepare=True, binary=True)

        return [int(i[0]) for i in result.fetchall()]
