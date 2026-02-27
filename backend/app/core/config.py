from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    app_name: str = "TrueFaceSwapVideo API"
    env: str = os.getenv("ENV", "dev")
    api_prefix: str = "/v1"
    cors_origins_raw: str = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")

    data_root: Path = Path(os.getenv("DATA_ROOT", "../data")).resolve()
    uploads_dir_name: str = "uploads"
    jobs_dir_name: str = "jobs"
    outputs_dir_name: str = "outputs"

    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./truefaceswap.db")

    public_base_url: str = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000")
    asset_token_secret: str = os.getenv("ASSET_TOKEN_SECRET", "change-me-asset-secret")
    asset_token_ttl_seconds: int = int(os.getenv("ASSET_TOKEN_TTL_SECONDS", "900"))

    runpod_enabled: bool = os.getenv("RUNPOD_ENABLED", "true").lower() == "true"
    runpod_api_key: str = os.getenv("RUNPOD_API_KEY", "")
    runpod_endpoint_id: str = os.getenv("RUNPOD_ENDPOINT_ID", "")
    runpod_api_base: str = os.getenv("RUNPOD_API_BASE", "https://api.runpod.ai/v2")

    callback_secret: str = os.getenv("CALLBACK_SECRET", "change-me-callback-secret")

    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "500"))
    allowed_video_ext: tuple[str, ...] = (".mp4", ".mov", ".mkv", ".webm")
    allowed_image_ext: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")
    allowed_audio_ext: tuple[str, ...] = (".wav", ".mp3", ".m4a", ".aac")


settings = Settings()


def cors_origins() -> list[str]:
    return [origin.strip() for origin in settings.cors_origins_raw.split(",") if origin.strip()]


def ensure_data_dirs() -> None:
    for name in (
        settings.uploads_dir_name,
        settings.jobs_dir_name,
        settings.outputs_dir_name,
    ):
        (settings.data_root / name).mkdir(parents=True, exist_ok=True)
