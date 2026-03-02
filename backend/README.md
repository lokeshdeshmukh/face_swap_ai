# Backend (FastAPI + SQLite + Local/S3 Storage + Runpod Serverless)

## Run

```bash
cd backend
python3.13 -m venv .venv313
source .venv313/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

## Storage modes

- `STORAGE_BACKEND=local`:
  - assets are served from `/v1/assets/{token}`
  - requires `PUBLIC_BASE_URL` reachable by Runpod
- `STORAGE_BACKEND=s3`:
  - assets/outputs are stored in S3 and shared via presigned URLs
  - set `AWS_PROFILE`, `S3_REGION`, `S3_BUCKET`, `S3_PREFIX`

## API

- `POST /v1/jobs`
  - required form fields: `mode`
  - generation modes:
    - `portrait_reenactment`
    - required driving input: `reference_video`
    - prompt is optional
    - optional text/number fields: `prompt`, `negative_prompt`, `motion_preset`, `style_preset`, `duration_seconds`, `seed`
    - `ai_video_generate`
    - `photo_to_video`
    - required text field: `prompt`
    - optional text/number fields: `negative_prompt`, `motion_preset`, `style_preset`, `duration_seconds`, `seed`
    - optional motion input: `reference_video`
  - legacy modes:
    - `video_swap`
    - `photo_sing`
    - required file field: `reference_video`
  - identity input options for all modes:
    - `source_images` (repeatable file field; one or more images)
    - `source_video` (optional video; frames are sampled in worker)
    - legacy `source_image` (single file) remains supported
  - optional: `driving_audio`
- `GET /v1/jobs/{id}`
- `POST /v1/jobs/{id}/retry`
- `GET /v1/jobs/{id}/output`
- `GET /v1/assets/{token}`
- `POST /v1/runpod/callback`
