# TrueFaceSwapVideo (Local iMac + Runpod Serverless)

This repository implements a local-first POC with production-shaped boundaries:
- FastAPI backend (`backend/`)
- Next.js frontend (`frontend/`)
- Local SQLite + filesystem storage
- Runpod Serverless compute integration

## Implemented APIs

- `POST /v1/jobs`
- `GET /v1/jobs/{id}`
- `POST /v1/jobs/{id}/retry`
- `GET /v1/jobs/{id}/output`
- `GET /v1/assets/{token}`
- `POST /v1/runpod/callback`

## Local Run Steps

1. Backend
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

2. Frontend
```bash
cd frontend
npm install
NEXT_PUBLIC_API_BASE=http://localhost:8000 npm run dev
```

3. Expose backend to internet for Runpod callbacks/assets
- Use Cloudflare Tunnel (or equivalent) and set `PUBLIC_BASE_URL`.
- Example: `https://abc123.trycloudflare.com`

## Runpod Worker Contract

The backend calls Runpod with this payload shape:

```json
{
  "input": {
    "job_id": "<uuid>",
    "mode": "video_swap|photo_sing",
    "quality": "fast|balanced|max",
    "enable_4k": true,
    "aspect_ratio": "9:16|1:1|4:5",
    "assets": {
      "reference_video_url": "https://.../v1/assets/<token>",
      "source_image_url": "https://.../v1/assets/<token>",
      "driving_audio_url": "https://.../v1/assets/<token> (optional)"
    },
    "callback": {
      "url": "https://.../v1/runpod/callback",
      "secret": "<CALLBACK_SECRET>"
    }
  }
}
```

Runpod worker must callback with header `X-Callback-Signature: <hmac_sha256(body, callback_secret)>` and body:

```json
{
  "job_id": "<uuid>",
  "status": "completed|failed",
  "output_url": "https://.../result.mp4",
  "error": null
}
```

(or `output_base64` instead of `output_url`)

## Notes

- Queue is in-process (no Redis/Celery).
- Storage is local filesystem (no S3).
- Designed for later migration via provider boundaries (`StorageProvider`, `QueueProvider`, `ComputeProvider`).
# face_swap_ai
