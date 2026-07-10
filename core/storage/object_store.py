from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from datetime import timedelta
from functools import lru_cache
from typing import BinaryIO

try:
    from minio import Minio
except ImportError:  # API/mock profile can import without object-store extras
    Minio = None  # type: ignore[assignment,misc]


@dataclass(frozen=True)
class ObjectStoreConfig:
    endpoint: str = field(
        default_factory=lambda: os.getenv("MINIO_ENDPOINT", "localhost:9000")
    )
    access_key: str = field(default_factory=lambda: os.getenv("MINIO_USER", ""))
    secret_key: str = field(default_factory=lambda: os.getenv("MINIO_PASSWORD", ""))
    secure: bool = field(
        default_factory=lambda: os.getenv("MINIO_USE_SSL", "0").lower()
        in {"1", "true", "yes", "on"}
    )


class ObjectStore:
    """MinIO boundary; callers use bucket/object keys, never server paths."""

    allowed_buckets = frozenset({"uploads", "generated", "exports"})

    def __init__(self, config: ObjectStoreConfig | None = None) -> None:
        self.config = config or ObjectStoreConfig()
        if Minio is None:
            raise RuntimeError("MinIO client dependency is not installed")
        if not self.config.access_key or not self.config.secret_key:
            raise RuntimeError("MinIO credentials are required")
        self.client = Minio(
            self.config.endpoint,
            access_key=self.config.access_key,
            secret_key=self.config.secret_key,
            secure=self.config.secure,
        )

    def ensure_buckets(self) -> None:
        for bucket in self.allowed_buckets:
            if not self.client.bucket_exists(bucket):
                self.client.make_bucket(bucket)

    def put(
        self, bucket: str, key: str, stream: BinaryIO, size: int, content_type: str
    ) -> None:
        self._validate(bucket, key)
        self.client.put_object(bucket, key, stream, size, content_type=content_type)

    def put_bytes(
        self, bucket: str, key: str, payload: bytes, content_type: str
    ) -> None:
        self.put(bucket, key, io.BytesIO(payload), len(payload), content_type)

    def presigned_get(
        self, bucket: str, key: str, expires: timedelta = timedelta(minutes=15)
    ) -> str:
        self._validate(bucket, key)
        return self.client.presigned_get_object(bucket, key, expires=expires)

    def get_bytes(self, bucket: str, key: str) -> bytes:
        self._validate(bucket, key)
        response = self.client.get_object(bucket, key)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def delete(self, bucket: str, key: str) -> None:
        self._validate(bucket, key)
        self.client.remove_object(bucket, key)

    def _validate(self, bucket: str, key: str) -> None:
        if bucket not in self.allowed_buckets:
            raise ValueError(f"Unsupported object bucket: {bucket}")
        if not key or key.startswith("/") or ".." in key.split("/"):
            raise ValueError("Invalid object key")


@lru_cache(maxsize=1)
def get_object_store() -> ObjectStore:
    return ObjectStore()
