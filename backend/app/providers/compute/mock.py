from __future__ import annotations

from app.models.job import Job
from app.providers.compute.base import ComputeProvider


class MockComputeProvider(ComputeProvider):
    async def submit_job(
        self,
        job: Job,
        asset_urls: dict[str, str],
        callback_url: str,
        callback_secret: str,
    ) -> tuple[str | None, str | None]:
        # Local dev placeholder. Real inference should run on Runpod.
        return f"mock-{job.id}", job.id
