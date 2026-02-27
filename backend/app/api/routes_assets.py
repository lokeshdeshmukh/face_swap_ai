from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.core.container import container

router = APIRouter(prefix="/assets", tags=["assets"])


@router.get("/{token}")
def get_asset(token: str) -> FileResponse:
    try:
        path = container.storage_provider.resolve_asset_token(token)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    media_type = "application/octet-stream"
    ext = path.suffix.lower()
    if ext in {".mp4", ".mov", ".mkv", ".webm"}:
        media_type = "video/mp4"
    elif ext in {".jpg", ".jpeg", ".png", ".webp"}:
        media_type = "image/jpeg"
    elif ext in {".wav", ".mp3", ".m4a", ".aac"}:
        media_type = "audio/mpeg"

    return FileResponse(path=str(path), filename=path.name, media_type=media_type)
