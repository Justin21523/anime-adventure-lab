from __future__ import annotations

import pytest

from core.storage import object_store


class FakeMinio:
    def __init__(self, endpoint, **kwargs):
        self.endpoint = endpoint
        self.kwargs = kwargs


def test_object_store_reads_runtime_environment_and_validates_keys(monkeypatch):
    monkeypatch.setenv("MINIO_ENDPOINT", "minio:9000")
    monkeypatch.setenv("MINIO_USER", "demo")
    monkeypatch.setenv("MINIO_PASSWORD", "secret")
    monkeypatch.setattr(object_store, "Minio", FakeMinio)

    store = object_store.ObjectStore()

    assert store.client.endpoint == "minio:9000"
    store._validate("uploads", "worlds/demo/lore.txt")
    with pytest.raises(ValueError):
        store._validate("uploads", "../secret")
    with pytest.raises(ValueError):
        store._validate("unknown", "safe.txt")
