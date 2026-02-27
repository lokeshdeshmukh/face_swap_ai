from __future__ import annotations

import asyncio
import logging

from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.providers.queue.base import QueueProvider
from app.services.job_service import JobService

logger = logging.getLogger(__name__)


class InProcessQueueProvider(QueueProvider):
    def __init__(self, job_service: JobService) -> None:
        self.job_service = job_service
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        self._stopping.set()
        if self._task:
            await self._task
            self._task = None

    async def enqueue(self, job_id: str) -> None:
        await self.queue.put(job_id)

    async def _worker(self) -> None:
        while not self._stopping.is_set():
            try:
                job_id = await asyncio.wait_for(self.queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                db = SessionLocal()
                try:
                    await self.job_service.poll_inflight_jobs(db)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("inflight poll failed", extra={"error": str(exc)})
                finally:
                    db.close()
                continue

            db: Session = SessionLocal()
            try:
                await self.job_service.dispatch_to_compute(db, job_id)
            except Exception as exc:  # noqa: BLE001
                logger.exception("job dispatch failed", extra={"job_id": job_id, "error": str(exc)})
                job = self.job_service.get_job(db, job_id)
                if job:
                    job.status = "failed"
                    job.stage = "failed"
                    job.error_message = str(exc)
                    db.add(job)
                    db.commit()
            finally:
                db.close()
                self.queue.task_done()
