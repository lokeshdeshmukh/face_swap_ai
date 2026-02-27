# Runpod Worker

This worker expects the backend payload contract documented in the root README.

## Build

```bash
docker build -t truefaceswap-runpod-worker .
```

## Local Verification

Run a full local smoke test before deploying to Runpod:

```bash
bash scripts/verify-local-image.sh
```

## Deploy

Push the image to a registry and configure it as the Runpod Serverless endpoint image.

## Important

- This template assumes model CLIs are present in the image.
- For production-sized outputs, replace `output_base64` with object-storage upload and return an `output_url`.
- Build now pins FaceFusion via `FACEFUSION_REF` (default `3.5.3`) for reproducible images.
- Worker startup runs a dependency preflight:
  - Required: `ffmpeg`, `facefusion`, Python modules `requests`, `runpod`, `cv2`, `onnxruntime`.
  - Optional-by-default: `liveportrait`, `musetalk`, `realesrgan-ncnn-vulkan`.
- To make optional tools mandatory at startup, set env vars:
  - `REQUIRE_PHOTO_SING_DEPS=true`
  - `REQUIRE_4K_ENHANCER=true`
