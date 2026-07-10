from __future__ import annotations

from functools import lru_cache
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from core.config import get_config


class Database:
    """Owns the SQLAlchemy engine and short-lived session factory."""

    def __init__(self, url: str | None = None) -> None:
        resolved = url or get_config().database.url
        connect_args = (
            {"check_same_thread": False} if resolved.startswith("sqlite") else {}
        )
        self.engine: Engine = create_engine(
            resolved,
            pool_pre_ping=True,
            connect_args=connect_args,
        )
        self.session_factory = sessionmaker(
            bind=self.engine,
            class_=Session,
            expire_on_commit=False,
            autoflush=False,
        )

    @contextmanager
    def sessions(self) -> Iterator[Session]:
        with self.session_factory() as session:
            yield session


@lru_cache(maxsize=1)
def get_database() -> Database:
    return Database()
