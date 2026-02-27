from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class StorageProvider(ABC):
    @abstractmethod
    def persist_upload(self, job_id: str, file_name: str, content: bytes) -> Path:
        raise NotImplementedError

    @abstractmethod
    def output_path(self, job_id: str, file_name: str) -> Path:
        raise NotImplementedError

    @abstractmethod
    def build_asset_token(self, path: Path, ttl_seconds: int) -> str:
        raise NotImplementedError

    @abstractmethod
    def resolve_asset_token(self, token: str) -> Path:
        raise NotImplementedError
