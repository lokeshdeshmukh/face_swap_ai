# Runpod Worker

This worker expects the backend payload contract documented in the root README.

## Build

```bash
docker build -t truefaceswap-runpod-worker .
```

Generation-only worker image:

```bash
docker build -f Dockerfile.generation -t truefaceswap-generation-worker .
```

Concrete generation contract:

- `GENERATION_CONTRACT.md`

## Local Verification

Run a full local smoke test before deploying to Runpod:

```bash
bash scripts/verify-local-image.sh
```

Generation-only smoke test:

```bash
bash scripts/verify-local-image.sh truefaceswap-generation-worker:local generation
```

## Deploy

Push the image to a registry and configure it as the Runpod Serverless endpoint image.

## Important

- This template assumes model CLIs are present in the image.
- For production-sized outputs, replace `output_base64` with object-storage upload and return an `output_url`.
- Generation-first modes (`ai_video_generate`, `photo_to_video`) accept a `job_config` payload.
- Generation workers can use `Dockerfile.generation` with `WORKER_PIPELINE_MODE=generation`.
- Contract parsing/validation lives in `/worker/src/generation_contract.py`.
- To plug in a real in-house generation stack, implement the adapter CLI in `GENERATION_CONTRACT.md`.
- Default generation backend in this repo:
  - render: `python3 /worker/src/generation_render_cogvideox.py`
  - refine: `python3 /worker/src/generation_refine_basic.py`
- Default model:
  - `GENERATION_MODEL_ID=THUDM/CogVideoX-5b-I2V`
- Preferred split for staged pipelines:
  - `GENERATION_RENDER_COMMAND`
  - `GENERATION_REFINE_COMMAND`
- Backward-compatible fallback:
  - `GENERATION_PIPELINE_COMMAND` is treated like a render adapter and receives `--shot-plan ... --output ...`
- Required render adapter args:
  - `--shot-plan <json-path>`
  - `--output <video-path>`
  - optional `--report <json-path>`
- Required refine adapter args:
  - `--identity-pack <json-path>`
  - `--input <video-path>`
  - `--output <video-path>`
  - optional `--report <json-path>`
- Example local env for the generation image:
  - `GENERATION_RENDER_COMMAND="python3 /worker/scripts/example_generation_render.py"`
  - `GENERATION_REFINE_COMMAND="python3 /worker/scripts/example_generation_refine.py"`
- Useful generation env overrides:
  - `GENERATION_MODEL_ID`
  - `GENERATION_MODEL_DTYPE=auto|bf16|fp16`
  - `GENERATION_OFFLOAD_MODE=model|sequential|none`
  - `GENERATION_MULTI_IMAGE_MODE=hero_grid|primary_only`
  - `GENERATION_HERO_IMAGE_INDEX`
  - `GENERATION_NUM_INFERENCE_STEPS`
  - `GENERATION_NUM_FRAMES`
  - `GENERATION_OUTPUT_FPS`
  - `GENERATION_GUIDANCE_SCALE`
- If you explicitly point the generation worker back to the example adapters, output will be placeholder/demo quality.
- Build now pins FaceFusion via `FACEFUSION_REF` (default `3.5.3`) for reproducible images.
- Worker startup runs a dependency preflight:
  - Generation mode required: `ffmpeg`, Python modules `requests`, `runpod`, `torch`, `diffusers`, `transformers`, `accelerate`, `PIL`, plus CUDA available in `torch`.
  - Legacy mode required: `ffmpeg`, `facefusion`, Python modules `requests`, `runpod`, `cv2`, `onnxruntime`, `onnx`, `scipy`.
  - Optional-by-default: `liveportrait`, `musetalk`, `realesrgan-ncnn-vulkan`.
- Runtime download provider can be controlled with:
  - `FACEFUSION_DOWNLOAD_PROVIDERS="huggingface github"` (default)
  - Worker retries once with `github` automatically if model source validation fails.
- Video swap realism/speed can be tuned with:
  - `FACEFUSION_MODEL` (default by quality: `inswapper_128_fp16` for `fast`/`balanced`, `simswap_unofficial_512` for `max`)
  - `FACEFUSION_OUTPUT_VIDEO_ENCODER` (default: `h264_nvenc`)
  - `FACEFUSION_FACE_SWAPPER_WEIGHT` (balanced default: `0.85`, max default: `1.00`)
  - `FACEFUSION_FACE_SWAPPER_PIXEL_BOOST` (balanced default: `768x768`, max default: `1024x1024`)
  - `FACEFUSION_FACE_SELECTOR_MODE` (default: `reference`)
  - `FACEFUSION_REFERENCE_FACE_DISTANCE` (default: `0.45`)
  - `FACEFUSION_FACE_DETECTOR_ANGLES` (default: `0`)
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
