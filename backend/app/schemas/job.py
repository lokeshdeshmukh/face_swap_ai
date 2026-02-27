from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class JobMode(str, Enum):
    video_swap = "video_swap"
    photo_sing = "photo_sing"


class QualityTier(str, Enum):
    fast = "fast"
    balanced = "balanced"
    max = "max"


class AspectRatio(str, Enum):
    portrait = "9:16"
    square = "1:1"
    vertical = "4:5"


class JobCreateResponse(BaseModel):
    id: str
    status: str
    stage: str


class JobStatusResponse(BaseModel):
    id: str
    mode: JobMode
    quality: QualityTier
    enable_4k: bool
    aspect_ratio: AspectRatio
    status: str
    stage: str
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    request_id: str | None
    runpod_job_id: str | None
    error_message: str | None
    stage_timings: dict[str, dict[str, str]]
    output_url: str | None

class RunpodCallbackPayload(BaseModel):
    job_id: str
    status: str
    output_url: str | None = None
    output_base64: str | None = None
    error: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
