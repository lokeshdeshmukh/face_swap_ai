from __future__ import annotations

from abc import ABC, abstractmethod

from app.models.job import Job


class ComputeProvider(ABC):
    @abstractmethod
    async def submit_job(
        self,
        job: Job,
        asset_urls: dict[str, str],
        callback_url: str,
        callback_secret: str,
    ) -> tuple[str | None, str | None]:
        """Returns (runpod_job_id, request_id)."""
        raise NotImplementedError
