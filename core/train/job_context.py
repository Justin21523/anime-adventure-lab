from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass(frozen=True)
class JobContext:
    job_id: str
    job_type: str


_job_context: ContextVar[Optional[JobContext]] = ContextVar("job_context", default=None)


@contextmanager
def job_context(job_id: str, job_type: str) -> Iterator[JobContext]:
    ctx = JobContext(job_id=str(job_id).strip(), job_type=str(job_type).strip())
    token = _job_context.set(ctx)
    try:
        yield ctx
    finally:
        _job_context.reset(token)


def get_job_context() -> Optional[JobContext]:
    return _job_context.get()

