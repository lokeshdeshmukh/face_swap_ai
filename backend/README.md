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
- `GET /v1/jobs/{id}`
- `POST /v1/jobs/{id}/retry`
- `GET /v1/jobs/{id}/output`
- `GET /v1/assets/{token}`
- `POST /v1/runpod/callback`
