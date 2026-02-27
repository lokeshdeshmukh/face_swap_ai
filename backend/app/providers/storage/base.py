from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class StorageProvider(ABC):
    @abstractmethod
    def persist_upload(self, job_id: str, file_name: str, content: bytes) -> str:
        raise NotImplementedError

    @abstractmethod
    def persist_output(self, job_id: str, file_name: str, content: bytes) -> str:
        raise NotImplementedError

    @abstractmethod
    def build_asset_url(self, asset_ref: str, ttl_seconds: int) -> str:
        raise NotImplementedError

    def build_output_url(self, job_id: str, output_ref: str, ttl_seconds: int) -> str:
        return self.build_asset_url(output_ref, ttl_seconds)

    def build_worker_output_target(self, job_id: str, file_name: str, ttl_seconds: int) -> dict[str, str] | None:
        return None

    def resolve_asset_token(self, token: str) -> Path:
        raise ValueError("asset token route is only available for local storage")

    def resolve_output_path(self, output_ref: str) -> Path | None:
        return None
