from __future__ import annotations

from abc import ABC, abstractmethod


class QueueProvider(ABC):
    @abstractmethod
    async def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def stop(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def enqueue(self, job_id: str) -> None:
        raise NotImplementedError
