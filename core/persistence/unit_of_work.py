from __future__ import annotations

from types import TracebackType

from sqlalchemy.orm import Session

from .database import Database, get_database


class UnitOfWork:
    """Transaction boundary used by application services."""

    def __init__(self, database: Database | None = None) -> None:
        self.database = database or get_database()
        self.session: Session | None = None

    def __enter__(self) -> "UnitOfWork":
        self.session = self.database.session_factory()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.session is None:
            return
        try:
            if exc_type is None:
                self.session.commit()
            else:
                self.session.rollback()
        finally:
            self.session.close()

    def commit(self) -> None:
        if self.session is None:
            raise RuntimeError("UnitOfWork is not active")
        self.session.commit()

    def rollback(self) -> None:
        if self.session is not None:
            self.session.rollback()
