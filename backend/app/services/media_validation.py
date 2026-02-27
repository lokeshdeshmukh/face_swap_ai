from __future__ import annotations

from pathlib import Path

from app.core.config import settings


def validate_extension(path_or_name: Path | str, kind: str) -> None:
    ext = Path(path_or_name).suffix.lower()
    allowed = {
        "video": settings.allowed_video_ext,
        "image": settings.allowed_image_ext,
        "audio": settings.allowed_audio_ext,
    }[kind]
    if ext not in allowed:
        raise ValueError(f"unsupported {kind} extension: {ext}")
