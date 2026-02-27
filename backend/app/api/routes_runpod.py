from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.core.config import settings
from app.core.container import container
from app.schemas.job import RunpodCallbackPayload
from app.utils.signing import verify_webhook_signature

router = APIRouter(prefix="/runpod", tags=["runpod"])


@router.post("/callback")
async def runpod_callback(
    request: Request,
    db: Session = Depends(get_db),
    x_callback_signature: str | None = Header(default=None),
) -> dict[str, str]:
    body = await request.body()
    if not x_callback_signature:
        raise HTTPException(status_code=401, detail="missing callback signature")
    if not verify_webhook_signature(settings.callback_secret, body, x_callback_signature):
        raise HTTPException(status_code=401, detail="invalid callback signature")

    try:
        payload = RunpodCallbackPayload.model_validate_json(body)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.errors()) from exc

    try:
        job = await container.job_service.handle_callback(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"status": job.status, "job_id": job.id}
