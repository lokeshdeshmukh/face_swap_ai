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

    def persist_upload(self, job_id: str, file_name: str, content: bytes) -> str:
        job_dir = self.uploads_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        target = job_dir / file_name
        target.write_bytes(content)
        return str(target.resolve())

    def persist_output(self, job_id: str, file_name: str, content: bytes) -> str:
        job_dir = self.outputs_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        target = job_dir / file_name
        target.write_bytes(content)
        return str(target.resolve())

    def build_asset_url(self, asset_ref: str, ttl_seconds: int) -> str:
        path = Path(asset_ref)
        payload = {
            "path": str(path.resolve()),
            "exp": int(time.time()) + ttl_seconds,
        }
        token = self.signer.sign(payload)
        return f"{settings.public_base_url.rstrip('/')}{settings.api_prefix}/assets/{token}"

    def build_output_url(self, job_id: str, output_ref: str, ttl_seconds: int) -> str:
        return f"{settings.public_base_url.rstrip('/')}{settings.api_prefix}/jobs/{job_id}/output"

    def resolve_asset_token(self, token: str) -> Path:
        payload = self.signer.verify(token)
        path = Path(str(payload["path"]))
        if not path.exists():
            raise ValueError("asset path not found")
        return path

    def resolve_output_path(self, output_ref: str) -> Path | None:
        path = Path(output_ref)
        return path if path.exists() else None
