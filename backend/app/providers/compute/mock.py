from __future__ import annotations

from app.models.job import Job
from app.providers.compute.base import ComputeProvider


class MockComputeProvider(ComputeProvider):
    async def submit_job(
        self,
        job: Job,
        asset_urls: dict[str, str],
        output_target: dict[str, str] | None,
        callback_url: str | None,
        callback_secret: str | None,
    ) -> tuple[str | None, str | None]:
        # Local dev placeholder. Real inference should run on Runpod.
        return f"mock-{job.id}", job.id

    async def get_job_status(self, runpod_job_id: str) -> dict[str, object]:
        return {
            "id": runpod_job_id,
            "status": "IN_PROGRESS",
        }
