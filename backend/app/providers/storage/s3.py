from __future__ import annotations

from pathlib import Path

import boto3

from app.core.config import settings
from app.providers.storage.base import StorageProvider


class S3StorageProvider(StorageProvider):
    def __init__(self) -> None:
        if not settings.s3_bucket:
            raise RuntimeError("S3_BUCKET must be set when STORAGE_BACKEND=s3")

        session = boto3.Session(profile_name=settings.aws_profile, region_name=settings.s3_region)
        self.client = session.client("s3")
        self.bucket = settings.s3_bucket
        self.prefix = settings.s3_prefix.strip("/")

    def _key(self, *parts: str) -> str:
        clean = [p.strip("/") for p in parts if p]
        if self.prefix:
            clean.insert(0, self.prefix)
        return "/".join(clean)

    def persist_upload(self, job_id: str, file_name: str, content: bytes) -> str:
        key = self._key("uploads", job_id, Path(file_name).name)
        self.client.put_object(Bucket=self.bucket, Key=key, Body=content)
        return key

    def persist_output(self, job_id: str, file_name: str, content: bytes) -> str:
        key = self._key("outputs", job_id, Path(file_name).name)
        self.client.put_object(Bucket=self.bucket, Key=key, Body=content, ContentType="video/mp4")
        return key

    def build_asset_url(self, asset_ref: str, ttl_seconds: int) -> str:
        return self.client.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": self.bucket, "Key": asset_ref},
            ExpiresIn=ttl_seconds,
        )
