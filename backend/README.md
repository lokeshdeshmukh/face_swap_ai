# Backend (FastAPI + SQLite + Local Filesystem + Runpod Serverless)

## Run

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Required env vars

```bash
export PUBLIC_BASE_URL="https://your-cloudflare-tunnel.example.com"
export RUNPOD_ENABLED="true"
export RUNPOD_API_KEY="..."
export RUNPOD_ENDPOINT_ID="..."
export CALLBACK_SECRET="..."
export ASSET_TOKEN_SECRET="..."
```

`PUBLIC_BASE_URL` must be publicly reachable so Runpod can fetch `/v1/assets/{token}` and call `/v1/runpod/callback`.

## API

- `POST /v1/jobs`
- `GET /v1/jobs/{id}`
- `POST /v1/jobs/{id}/retry`
- `GET /v1/jobs/{id}/output`
- `GET /v1/assets/{token}`
- `POST /v1/runpod/callback`
