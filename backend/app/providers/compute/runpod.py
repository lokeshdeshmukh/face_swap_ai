from __future__ import annotations

import logging

import httpx

from app.core.config import settings
from app.models.job import Job
from app.providers.compute.base import ComputeProvider

logger = logging.getLogger(__name__)


class RunpodComputeProvider(ComputeProvider):
    async def submit_job(
        self,
        job: Job,
        asset_urls: dict[str, str],
        callback_url: str,
        callback_secret: str,
    ) -> tuple[str | None, str | None]:
        if not settings.runpod_api_key or not settings.runpod_endpoint_id:
            raise RuntimeError("RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID must be set")

        url = f"{settings.runpod_api_base.rstrip('/')}/{settings.runpod_endpoint_id}/run"
        payload = {
            "input": {
                "job_id": job.id,
                "mode": job.mode,
                "quality": job.quality,
                "enable_4k": job.enable_4k,
                "aspect_ratio": job.aspect_ratio,
                "assets": asset_urls,
                "callback": {
                    "url": callback_url,
                    "secret": callback_secret,
                },
            }
        }
        headers = {
            "Authorization": f"Bearer {settings.runpod_api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=45) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            body = response.json()

        runpod_job_id = body.get("id")
        request_id = None
        if isinstance(body.get("input"), dict):
            request_id = body["input"].get("job_id")
        logger.info("submitted to runpod", extra={"job_id": job.id, "runpod_job_id": runpod_job_id})
        return runpod_job_id, request_id
