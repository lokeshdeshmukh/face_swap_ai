from __future__ import annotations

from pathlib import Path

from app.core.config import settings


def validate_extension(path: Path, kind: str) -> None:
    ext = path.suffix.lower()
    allowed = {
        "video": settings.allowed_video_ext,
        "image": settings.allowed_image_ext,
        "audio": settings.allowed_audio_ext,
    }[kind]
    if ext not in allowed:
        raise ValueError(f"unsupported {kind} extension: {ext}")


def validate_file_size(path: Path) -> None:
    max_size = settings.max_upload_mb * 1024 * 1024
    if path.stat().st_size > max_size:
        raise ValueError(f"file exceeds max size: {settings.max_upload_mb}MB")
