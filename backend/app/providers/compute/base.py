from __future__ import annotations

from abc import ABC, abstractmethod

from app.models.job import Job


class ComputeProvider(ABC):
    @abstractmethod
    async def submit_job(
        self,
        job: Job,
        asset_urls: dict[str, str],
        output_target: dict[str, str] | None,
        callback_url: str | None,
        callback_secret: str | None,
    ) -> tuple[str | None, str | None]:
        """Returns (runpod_job_id, request_id)."""
        raise NotImplementedError

    @abstractmethod
    async def get_job_status(self, runpod_job_id: str) -> dict[str, object]:
        raise NotImplementedError
