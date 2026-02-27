from __future__ import annotations

from app.core.config import settings
from app.providers.compute.mock import MockComputeProvider
from app.providers.compute.runpod import RunpodComputeProvider
from app.providers.queue.inprocess import InProcessQueueProvider
from app.providers.storage.local import LocalStorageProvider
from app.services.job_service import JobService


class Container:
    def __init__(self) -> None:
        self.storage_provider = LocalStorageProvider()
        self.compute_provider = RunpodComputeProvider() if settings.runpod_enabled else MockComputeProvider()
        self.job_service = JobService(self.storage_provider, self.compute_provider)
        self.queue_provider = InProcessQueueProvider(self.job_service)


container = Container()
