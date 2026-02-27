from __future__ import annotations

import unittest
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.base import Base
from app.models.job import Job
from app.providers.compute.base import ComputeProvider
from app.providers.storage.base import StorageProvider
from app.services.job_service import JobService


class _DummyStorage(StorageProvider):
    def persist_upload(self, job_id: str, file_name: str, content: bytes) -> str:
        return f"uploads/{job_id}/{file_name}"

    def persist_output(self, job_id: str, file_name: str, content: bytes) -> str:
        return f"outputs/{job_id}/{file_name}"

    def build_asset_url(self, asset_ref: str, ttl_seconds: int) -> str:
        return f"https://example.test/{asset_ref}?ttl={ttl_seconds}"


class _DummyCompute(ComputeProvider):
    async def submit_job(self, job, asset_urls, output_target, callback_url, callback_secret):  # type: ignore[no-untyped-def]
        return None, None

    async def get_job_status(self, runpod_job_id: str) -> dict[str, object]:
        return {"status": "IN_PROGRESS"}


class TestJobService(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine, future=True)
        self.service = JobService(storage=_DummyStorage(), compute=_DummyCompute())

    def test_list_existing_by_hash_handles_duplicates(self) -> None:
        db: Session = self.SessionLocal()
        try:
            base = datetime(2026, 2, 27, 12, 0, 0)
            old = Job(
                id="11111111-1111-1111-1111-111111111111",
                mode="video_swap",
                quality="balanced",
                enable_4k=False,
                aspect_ratio="9:16",
                config_hash="same-hash",
                status="failed",
                stage="failed",
                stage_timings_json="{}",
                reference_video_path="uploads/a/ref.mp4",
                source_image_path="uploads/a/src.jpg",
                created_at=base,
                updated_at=base,
            )
            new = Job(
                id="22222222-2222-2222-2222-222222222222",
                mode="video_swap",
                quality="balanced",
                enable_4k=False,
                aspect_ratio="9:16",
                config_hash="same-hash",
                status="done",
                stage="done",
                stage_timings_json="{}",
                reference_video_path="uploads/b/ref.mp4",
                source_image_path="uploads/b/src.jpg",
                created_at=base + timedelta(seconds=10),
                updated_at=base + timedelta(seconds=10),
            )
            db.add_all([old, new])
            db.commit()

            result = self.service.list_existing_by_hash(db, "same-hash")
            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result.id, new.id)
        finally:
            db.close()


if __name__ == "__main__":
    unittest.main()
