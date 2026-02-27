from __future__ import annotations

import time
from pathlib import Path

from app.core.config import settings
from app.providers.storage.base import StorageProvider
from app.utils.signing import TokenSigner


class LocalStorageProvider(StorageProvider):
    def __init__(self) -> None:
        self.uploads_root = settings.data_root / settings.uploads_dir_name
        self.outputs_root = settings.data_root / settings.outputs_dir_name
        self.signer = TokenSigner(settings.asset_token_secret)

    def persist_upload(self, job_id: str, file_name: str, content: bytes) -> Path:
        job_dir = self.uploads_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        target = job_dir / file_name
        target.write_bytes(content)
        return target

    def output_path(self, job_id: str, file_name: str) -> Path:
        job_dir = self.outputs_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir / file_name

    def build_asset_token(self, path: Path, ttl_seconds: int) -> str:
        payload = {
            "path": str(path.resolve()),
            "exp": int(time.time()) + ttl_seconds,
        }
        return self.signer.sign(payload)

    def resolve_asset_token(self, token: str) -> Path:
        payload = self.signer.verify(token)
        path = Path(str(payload["path"]))
        if not path.exists():
            raise ValueError("asset path not found")
        return path
