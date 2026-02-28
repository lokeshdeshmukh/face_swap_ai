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
  - Required: `ffmpeg`, `facefusion`, Python modules `requests`, `runpod`, `cv2`, `onnxruntime`, `onnx`, `scipy`.
  - Optional-by-default: `liveportrait`, `musetalk`, `realesrgan-ncnn-vulkan`.
- Runtime download provider can be controlled with:
  - `FACEFUSION_DOWNLOAD_PROVIDERS="huggingface github"` (default)
  - Worker retries once with `github` automatically if model source validation fails.
- Video swap realism/speed can be tuned with:
  - `FACEFUSION_MODEL` (default by quality: `inswapper_128_fp16` for `fast`/`balanced`, `simswap_unofficial_512` for `max`)
  - `FACEFUSION_OUTPUT_VIDEO_ENCODER` (default: `h264_nvenc`)
  - `FACEFUSION_FACE_SWAPPER_WEIGHT` (balanced default: `0.85`, max default: `1.00`)
  - `FACEFUSION_FACE_SWAPPER_PIXEL_BOOST` (balanced default: `768x768`, max default: `1024x1024`)
  - You can override by quality using suffixes:
    - `FACEFUSION_MODEL_BALANCED`, `FACEFUSION_MODEL_MAX`
    - `FACEFUSION_OUTPUT_VIDEO_ENCODER_BALANCED`, `FACEFUSION_OUTPUT_VIDEO_ENCODER_MAX`
    - `FACEFUSION_FACE_SWAPPER_WEIGHT_BALANCED`, `FACEFUSION_FACE_SWAPPER_WEIGHT_MAX`
    - `FACEFUSION_FACE_SWAPPER_PIXEL_BOOST_BALANCED`, `FACEFUSION_FACE_SWAPPER_PIXEL_BOOST_MAX`
- Adaptive selector probe can be tuned with:
  - `FACEFUSION_ADAPTIVE_SELECTOR=true` (default)
  - `FACEFUSION_PROBE_FRAMES=72` (default)
  - `FACEFUSION_SELECTOR_CANDIDATES="one reference many"` (default)
- To make optional tools mandatory at startup, set env vars:
  - `REQUIRE_PHOTO_SING_DEPS=true`
  - `REQUIRE_4K_ENHANCER=true`
