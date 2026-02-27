from __future__ import annotations

import base64
import hashlib
import json
import logging
import uuid
from datetime import datetime

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.job import Job
from app.providers.compute.base import ComputeProvider
from app.providers.storage.base import StorageProvider
from app.schemas.job import AspectRatio, JobMode, QualityTier, RunpodCallbackPayload
from app.services.media_validation import validate_extension
from app.utils.hash_utils import stable_config_hash

logger = logging.getLogger(__name__)


class JobService:
    def __init__(self, storage: StorageProvider, compute: ComputeProvider) -> None:
        self.storage = storage
        self.compute = compute

    def get_job(self, db: Session, job_id: str) -> Job | None:
        return db.get(Job, job_id)

    def list_existing_by_hash(self, db: Session, config_hash: str) -> Job | None:
        # Multiple historical rows can exist for the same config_hash.
        # Return the newest match deterministically instead of raising.
        stmt = (
            select(Job)
            .where(Job.config_hash == config_hash)
            .order_by(Job.created_at.desc(), Job.id.desc())
            .limit(1)
        )
        return db.execute(stmt).scalars().first()

    def create_job(
        self,
        db: Session,
        mode: JobMode,
        quality: QualityTier,
        enable_4k: bool,
        aspect_ratio: AspectRatio,
        reference_video_name: str,
        reference_video_bytes: bytes,
        source_image_name: str,
        source_image_bytes: bytes,
        driving_audio_name: str | None,
        driving_audio_bytes: bytes | None,
    ) -> Job:
        job_id = str(uuid.uuid4())

        validate_extension(reference_video_name, "video")
        validate_extension(source_image_name, "image")
        if driving_audio_name:
            validate_extension(driving_audio_name, "audio")

        self._validate_size(reference_video_bytes)
        self._validate_size(source_image_bytes)
        if driving_audio_bytes:
            self._validate_size(driving_audio_bytes)

        config_hash = stable_config_hash(
            [
                mode.value,
                quality.value,
                str(enable_4k),
                aspect_ratio.value,
                self._sha256_bytes(reference_video_bytes),
                self._sha256_bytes(source_image_bytes),
                self._sha256_bytes(driving_audio_bytes) if driving_audio_bytes else "",
            ]
        )

        existing = self.list_existing_by_hash(db, config_hash)
        if existing and existing.status in {"queued", "processing", "done"}:
            return existing

        reference_video_path = self.storage.persist_upload(job_id, reference_video_name, reference_video_bytes)
        source_image_path = self.storage.persist_upload(job_id, source_image_name, source_image_bytes)
        driving_audio_path = None
        if driving_audio_name and driving_audio_bytes:
            driving_audio_path = self.storage.persist_upload(job_id, driving_audio_name, driving_audio_bytes)

        now = datetime.utcnow()
        timings = {
            "queued": {
                "start": now.isoformat(),
                "end": now.isoformat(),
            }
        }
        job = Job(
            id=job_id,
            mode=mode.value,
            quality=quality.value,
            enable_4k=enable_4k,
            aspect_ratio=aspect_ratio.value,
            config_hash=config_hash,
            status="queued",
            stage="queued",
            stage_timings_json=json.dumps(timings),
            reference_video_path=reference_video_path,
            source_image_path=source_image_path,
            driving_audio_path=driving_audio_path,
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        return job

    def _start_stage(self, job: Job, stage: str, status: str) -> None:
        now = datetime.utcnow().isoformat()
        timings = json.loads(job.stage_timings_json or "{}")
        timings.setdefault(stage, {})
        timings[stage]["start"] = timings[stage].get("start", now)
        timings[stage]["end"] = now

        job.stage = stage
        job.status = status
        if not job.started_at:
            job.started_at = datetime.utcnow()
        job.stage_timings_json = json.dumps(timings)
        job.updated_at = datetime.utcnow()

    async def dispatch_to_compute(self, db: Session, job_id: str) -> None:
        job = db.get(Job, job_id)
        if not job:
            return
        if job.status not in {"queued", "retry"}:
            return

        self._start_stage(job, "preprocessing", "processing")
        db.add(job)
        db.commit()

        asset_urls = {
            "reference_video_url": self.storage.build_asset_url(
                job.reference_video_path, settings.asset_token_ttl_seconds
            ),
            "source_image_url": self.storage.build_asset_url(job.source_image_path, settings.asset_token_ttl_seconds),
        }
        if job.driving_audio_path:
            asset_urls["driving_audio_url"] = self.storage.build_asset_url(
                job.driving_audio_path, settings.asset_token_ttl_seconds
            )

        callback_url = self._callback_url_or_none()
        callback_secret = settings.callback_secret if callback_url else None

        self._start_stage(job, "generating", "processing")
        db.add(job)
        db.commit()

        runpod_job_id, request_id = await self.compute.submit_job(
            job=job,
            asset_urls=asset_urls,
            callback_url=callback_url,
            callback_secret=callback_secret,
        )

        job.runpod_job_id = runpod_job_id
        job.request_id = request_id
        db.add(job)
        db.commit()

    async def poll_inflight_jobs(self, db: Session) -> None:
        stmt = select(Job).where(
            Job.status == "processing",
            Job.runpod_job_id.is_not(None),
        )
        jobs = db.execute(stmt).scalars().all()
        for job in jobs:
            if not job.runpod_job_id:
                continue
            await self._reconcile_runpod_status(db, job)

    async def _reconcile_runpod_status(self, db: Session, job: Job) -> None:
        if not job.runpod_job_id:
            return

        status_body = await self.compute.get_job_status(job.runpod_job_id)
        run_status = str(status_body.get("status", "")).upper()

        if run_status in {"IN_QUEUE", "IN_PROGRESS"}:
            return

        if run_status in {"FAILED", "CANCELLED", "TIMED_OUT", "NOT_FOUND"}:
            self._start_stage(job, "failed", "failed")
            job.error_message = str(status_body.get("error") or "runpod job failed")
            job.finished_at = datetime.utcnow()
            db.add(job)
            db.commit()
            return

        if run_status != "COMPLETED":
            return

        output = status_body.get("output")
        if not isinstance(output, dict):
            self._start_stage(job, "failed", "failed")
            job.error_message = "runpod completed without output payload"
            job.finished_at = datetime.utcnow()
            db.add(job)
            db.commit()
            return

        if str(output.get("status", "")).lower() in {"failed", "error"}:
            self._start_stage(job, "failed", "failed")
            job.error_message = str(output.get("error") or "worker failed")
            job.finished_at = datetime.utcnow()
            db.add(job)
            db.commit()
            return

        payload = RunpodCallbackPayload(
            job_id=job.id,
            status="completed",
            output_url=output.get("output_url"),
            output_base64=output.get("output_base64"),
            metadata={"source": "runpod-status-poll"},
        )
        await self.handle_callback(db, payload)

    async def handle_callback(self, db: Session, payload: RunpodCallbackPayload) -> Job:
        job = db.get(Job, payload.job_id)
        if not job:
            raise ValueError("job not found")

        # Idempotency for duplicate callback deliveries.
        if job.status == "done" and job.output_path:
            return job

        if payload.status.lower() in {"failed", "error"}:
            self._start_stage(job, "failed", "failed")
            job.error_message = payload.error or "runpod worker failed"
            job.finished_at = datetime.utcnow()
            db.add(job)
            db.commit()
            db.refresh(job)
            return job

        self._start_stage(job, "enhancing", "processing")
        db.add(job)
        db.commit()

        output_bytes: bytes
        if payload.output_base64:
            output_bytes = base64.b64decode(payload.output_base64)
        elif payload.output_url:
            output_bytes = await self._download_to_bytes(payload.output_url)
        else:
            raise ValueError("callback missing output_base64 or output_url")

        output_ref = self.storage.persist_output(job.id, "result.mp4", output_bytes)

        self._start_stage(job, "packaging", "processing")
        self._start_stage(job, "done", "done")
        job.output_path = output_ref
        job.error_message = None
        job.finished_at = datetime.utcnow()
        db.add(job)
        db.commit()
        db.refresh(job)
        return job

    async def _download_to_bytes(self, url: str) -> bytes:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.content

    def build_output_url(self, job: Job) -> str | None:
        if not job.output_path:
            return None
        return self.storage.build_output_url(job.id, job.output_path, settings.output_url_ttl_seconds)

    @staticmethod
    def _sha256_bytes(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    @staticmethod
    def _validate_size(content: bytes) -> None:
        max_size = settings.max_upload_mb * 1024 * 1024
        if len(content) > max_size:
            raise ValueError(f"file exceeds max size: {settings.max_upload_mb}MB")

    @staticmethod
    def _callback_url_or_none() -> str | None:
        raw = settings.public_base_url.strip()
        if not raw:
            return None
        if "your-tunnel-domain.example.com" in raw:
            return None
        if raw.startswith("http://localhost") or raw.startswith("http://127.0.0.1"):
            return None
        return f"{raw.rstrip('/')}{settings.api_prefix}/runpod/callback"
