from __future__ import annotations

import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.core.container import container
from app.schemas.job import AspectRatio, JobCreateResponse, JobMode, JobStatusResponse, QualityTier

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("", response_model=JobCreateResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_job(
    mode: JobMode = Form(...),
    quality: QualityTier = Form(QualityTier.balanced),
    enable_4k: bool = Form(False),
    aspect_ratio: AspectRatio = Form(AspectRatio.portrait),
    reference_video: UploadFile = File(...),
    source_image: UploadFile = File(...),
    driving_audio: UploadFile | None = File(None),
    db: Session = Depends(get_db),
) -> JobCreateResponse:
    reference_video_bytes = await reference_video.read()
    source_image_bytes = await source_image.read()
    driving_audio_bytes = await driving_audio.read() if driving_audio else None

    try:
        job = container.job_service.create_job(
            db=db,
            mode=mode,
            quality=quality,
            enable_4k=enable_4k,
            aspect_ratio=aspect_ratio,
            reference_video_name=reference_video.filename or "reference.mp4",
            reference_video_bytes=reference_video_bytes,
            source_image_name=source_image.filename or "source.jpg",
            source_image_bytes=source_image_bytes,
            driving_audio_name=driving_audio.filename if driving_audio else None,
            driving_audio_bytes=driving_audio_bytes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if job.status == "queued":
        await container.queue_provider.enqueue(job.id)

    return JobCreateResponse(id=job.id, status=job.status, stage=job.stage)


@router.post("/{job_id}/retry", response_model=JobCreateResponse)
async def retry_job(job_id: str, db: Session = Depends(get_db)) -> JobCreateResponse:
    job = container.job_service.get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    if job.status not in {"failed"}:
        raise HTTPException(status_code=400, detail="only failed jobs can be retried")

    job.status = "retry"
    job.stage = "queued"
    job.error_message = None
    db.add(job)
    db.commit()
    db.refresh(job)

    await container.queue_provider.enqueue(job.id)
    return JobCreateResponse(id=job.id, status=job.status, stage=job.stage)


@router.get("/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str, db: Session = Depends(get_db)) -> JobStatusResponse:
    job = container.job_service.get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    timings = json.loads(job.stage_timings_json or "{}")
    return JobStatusResponse(
        id=job.id,
        mode=job.mode,
        quality=job.quality,
        enable_4k=job.enable_4k,
        aspect_ratio=job.aspect_ratio,
        status=job.status,
        stage=job.stage,
        created_at=job.created_at,
        updated_at=job.updated_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        request_id=job.request_id,
        runpod_job_id=job.runpod_job_id,
        error_message=job.error_message,
        stage_timings=timings,
        output_url=container.job_service.build_output_url(job),
    )


@router.get("/{job_id}/output")
def get_job_output(job_id: str, db: Session = Depends(get_db)) -> FileResponse:
    job = container.job_service.get_job(db, job_id)
    if not job or not job.output_path:
        raise HTTPException(status_code=404, detail="output not available")

    path = container.storage_provider.resolve_output_path(job.output_path)
    if path is None:
        raise HTTPException(status_code=404, detail="output is not stored locally")
    if not path.exists():
        raise HTTPException(status_code=404, detail="output file missing")

    return FileResponse(path=str(path), filename=f"{job.id}.mp4", media_type="video/mp4")
