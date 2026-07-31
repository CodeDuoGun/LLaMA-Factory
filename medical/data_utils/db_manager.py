# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""基于 SQLAlchemy 连接池的通用数据库管理器."""

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from sqlalchemy import MetaData, Table, create_engine, delete, func, insert, select, text, update
from sqlalchemy.engine import Connection, Engine, Result
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import StaticPool

from medical.data_utils.config import config


class DBManager:
    """提供连接管理及数据表通用 CRUD.

    Args:
        database_url: SQLAlchemy 数据库连接地址。为空时读取 config.yaml 的
            ``DATABASE_URL``。
        pool_size: 连接池常驻连接数。
        max_overflow: 超出 ``pool_size`` 后允许临时创建的连接数。
        pool_timeout: 获取连接的最长等待秒数。
        pool_recycle: 连接回收秒数，避免 MySQL 服务端断开空闲连接。
        echo: 是否输出 SQLAlchemy SQL 日志。
    """

    def __init__(
        self,
        database_url: str | None = None,
        *,
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_timeout: int = 30,
        pool_recycle: int = 3600,
        echo: bool = False,
    ) -> None:
        url = database_url or config.get("DATABASE_URL")
        if not url:
            raise ValueError("未配置 DATABASE_URL")

        engine_options: dict[str, Any] = {
            "echo": echo,
            "pool_pre_ping": True,
        }
        if url.startswith("sqlite"):
            # 内存 SQLite 必须复用同一连接，主要方便测试和本地使用。
            if ":memory:" in url:
                engine_options.update(poolclass=StaticPool, connect_args={"check_same_thread": False})
        else:
            engine_options.update(
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_timeout=pool_timeout,
                pool_recycle=pool_recycle,
            )

        self.engine: Engine = create_engine(url, **engine_options)
        self._metadata = MetaData()
        self._tables: dict[tuple[str | None, str], Table] = {}

    @contextmanager
    def connection(self, *, transactional: bool = False) -> Iterator[Connection]:
        """获取并在使用后释放连接.

        ``transactional=True`` 时，正常退出自动提交，发生异常自动回滚。
        连接退出上下文后会归还连接池，而不是直接销毁。
        """
        context = self.engine.begin() if transactional else self.engine.connect()
        with context as connection:
            yield connection

    def get_connection(self) -> Connection:
        """从连接池获取连接，调用方应使用 ``release_connection`` 释放."""
        return self.engine.connect()

    @staticmethod
    def release_connection(connection: Connection) -> None:
        """将连接归还连接池."""
        connection.close()

    def close(self) -> None:
        """关闭连接池中的全部连接."""
        self.engine.dispose()
        self._tables.clear()
        self._metadata.clear()

    def ping(self) -> bool:
        """检查数据库是否可连接."""
        try:
            with self.connection() as connection:
                connection.execute(text("SELECT 1"))
            return True
        except SQLAlchemyError:
            return False

    def refresh_table(self, table_name: str, *, schema: str | None = None) -> Table:
        """重新从数据库加载表结构并更新缓存."""
        key = (schema, table_name)
        old_table = self._tables.pop(key, None)
        if old_table is not None and old_table.metadata is self._metadata:
            self._metadata.remove(old_table)
        return self._get_table(table_name, schema=schema)

    def create_table(self, table: Table, *, checkfirst: bool = True) -> None:
        """创建由 SQLAlchemy ``Table`` 定义的数据表."""
        table.create(bind=self.engine, checkfirst=checkfirst)
        self._cache_table(table)

    def drop_table(self, table_name: str, *, schema: str | None = None, checkfirst: bool = True) -> None:
        """删除数据表."""
        key = (schema, table_name)
        table = self._tables.get(key)
        if table is None:
            table = Table(table_name, MetaData(), schema=schema)
        table.drop(bind=self.engine, checkfirst=checkfirst)
        self._tables.pop(key, None)
        if table.metadata is self._metadata:
            self._metadata.remove(table)

    def insert(self, table_name: str, data: Mapping[str, Any], *, schema: str | None = None) -> Any:
        """插入单条数据，返回数据库生成的主键（没有主键时返回 ``None``）."""
        table = self._get_table(table_name, schema=schema)
        self._validate_columns(table, data)
        with self.connection(transactional=True) as connection:
            result = connection.execute(insert(table).values(**dict(data)))
            return result.inserted_primary_key[0] if result.inserted_primary_key else None

    def insert_many(self, table_name: str, rows: Sequence[Mapping[str, Any]], *, schema: str | None = None) -> int:
        """批量插入数据，返回受影响行数."""
        if not rows:
            return 0
        table = self._get_table(table_name, schema=schema)
        for row in rows:
            self._validate_columns(table, row)
        with self.connection(transactional=True) as connection:
            result = connection.execute(insert(table), [dict(row) for row in rows])
            return self._rowcount(result)

    def select(
        self,
        table_name: str,
        *,
        filters: Mapping[str, Any] | None = None,
        columns: Sequence[str] | None = None,
        limit: int | None = None,
        offset: int = 0,
        order_by: Sequence[str] | None = None,
        schema: str | None = None,
    ) -> list[dict[str, Any]]:
        """查询数据，过滤条件为字段等值匹配，结果返回字典列表.

        ``order_by`` 中字段名前加 ``-`` 表示降序，例如 ``["-id"]``。
        """
        table = self._get_table(table_name, schema=schema)
        selected_columns = [self._get_column(table, name) for name in columns] if columns else [table]
        statement = select(*selected_columns)
        statement = self._apply_filters(statement, table, filters)

        for field in order_by or ():
            descending = field.startswith("-")
            column = self._get_column(table, field[1:] if descending else field)
            statement = statement.order_by(column.desc() if descending else column.asc())
        if limit is not None:
            if limit < 0:
                raise ValueError("limit 不能小于 0")
            statement = statement.limit(limit)
        if offset < 0:
            raise ValueError("offset 不能小于 0")
        if offset:
            statement = statement.offset(offset)

        with self.connection() as connection:
            return [dict(row) for row in connection.execute(statement).mappings().all()]

    def select_one(
        self,
        table_name: str,
        *,
        filters: Mapping[str, Any] | None = None,
        columns: Sequence[str] | None = None,
        order_by: Sequence[str] | None = None,
        schema: str | None = None,
    ) -> dict[str, Any] | None:
        """查询单条数据，不存在时返回 ``None``."""
        rows = self.select(
            table_name,
            filters=filters,
            columns=columns,
            limit=1,
            order_by=order_by,
            schema=schema,
        )
        return rows[0] if rows else None

    def update(
        self,
        table_name: str,
        data: Mapping[str, Any],
        *,
        filters: Mapping[str, Any] | None,
        schema: str | None = None,
        allow_all: bool = False,
    ) -> int:
        """更新数据并返回受影响行数；默认禁止无条件更新."""
        if not filters and not allow_all:
            raise ValueError("无条件更新可能影响全表，请设置 allow_all=True 明确允许")
        if not data:
            return 0
        table = self._get_table(table_name, schema=schema)
        self._validate_columns(table, data)
        statement = self._apply_filters(update(table).values(**dict(data)), table, filters)
        with self.connection(transactional=True) as connection:
            return self._rowcount(connection.execute(statement))

    def delete(
        self,
        table_name: str,
        *,
        filters: Mapping[str, Any] | None,
        schema: str | None = None,
        allow_all: bool = False,
    ) -> int:
        """删除数据并返回受影响行数；默认禁止无条件删除."""
        if not filters and not allow_all:
            raise ValueError("无条件删除可能清空全表，请设置 allow_all=True 明确允许")
        table = self._get_table(table_name, schema=schema)
        statement = self._apply_filters(delete(table), table, filters)
        with self.connection(transactional=True) as connection:
            return self._rowcount(connection.execute(statement))

    def count(self, table_name: str, *, filters: Mapping[str, Any] | None = None, schema: str | None = None) -> int:
        """统计符合条件的数据行数."""
        table = self._get_table(table_name, schema=schema)
        statement = self._apply_filters(select(func.count()).select_from(table), table, filters)
        with self.connection() as connection:
            return int(connection.execute(statement).scalar_one())

    def _get_table(self, table_name: str, *, schema: str | None = None) -> Table:
        key = (schema, table_name)
        if key not in self._tables:
            self._tables[key] = Table(table_name, self._metadata, schema=schema, autoload_with=self.engine)
        return self._tables[key]

    def _cache_table(self, table: Table) -> None:
        key = (table.schema, table.name)
        existing = self._tables.get(key)
        if existing is not None and existing is not table and existing.metadata is self._metadata:
            self._metadata.remove(existing)
        self._tables[key] = table

    @staticmethod
    def _get_column(table: Table, name: str) -> Any:
        if name not in table.c:
            raise ValueError(f"表 {table.fullname} 不存在字段: {name}")
        return table.c[name]

    @classmethod
    def _validate_columns(cls, table: Table, data: Mapping[str, Any]) -> None:
        for name in data:
            cls._get_column(table, name)

    @classmethod
    def _apply_filters(cls, statement: Any, table: Table, filters: Mapping[str, Any] | None) -> Any:
        for name, value in (filters or {}).items():
            column = cls._get_column(table, name)
            if value is None:
                statement = statement.where(column.is_(None))
            elif isinstance(value, (list, tuple, set, frozenset)):
                statement = statement.where(column.in_(value))
            else:
                statement = statement.where(column == value)
        return statement

    @staticmethod
    def _rowcount(result: Result[Any]) -> int:
        return max(result.rowcount or 0, 0)


# 默认实例：直接使用 config.yaml 中的 DATABASE_URL。
db_manager = DBManager()
