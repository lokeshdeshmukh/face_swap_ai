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

`Dockerfile.generation` is now optimized for the production `Full Body Reenactment` path. It intentionally omits the portrait runtime to keep build time and image size under control.

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
- Generation pipeline modes (`portrait_reenactment`, `ai_video_generate`, `photo_to_video`) accept a `job_config` payload.
- Generation workers can use `Dockerfile.generation` with `WORKER_PIPELINE_MODE=generation`.
- Contract parsing/validation lives in `/worker/src/generation_contract.py`.
- Driving-video reenactment now extracts a `control_bundle.json` before render.
- To plug in a real in-house generation stack, implement the adapter CLI in `GENERATION_CONTRACT.md`.
- Default generation backend in this repo:
  - portrait reenactment render: `python3 /worker/src/generation_render_reenactment.py`
  - portrait reenactment pipeline: not bundled in `Dockerfile.generation`
  - full body reenactment render: `python3 /worker/src/generation_render_full_body_reenactment.py`
  - full body reenactment pipeline: `python3 /worker/src/full_body_reenactment_mimicmotion.py`
  - render: `python3 /worker/src/generation_render_cogvideox.py`
  - refine: `python3 /worker/src/generation_refine_basic.py`
- Portrait reenactment requires a dedicated in-repo model runner:
  - this lean image leaves `PORTRAIT_REENACTMENT_PIPELINE_COMMAND` unset
  - portrait mode will fail cleanly on this endpoint unless you supply a separate portrait backend image/runtime
- Full Body Reenactment is a separate product path:
  - default render wrapper: `FULL_BODY_REENACTMENT_RENDER_COMMAND="python3 /worker/src/generation_render_full_body_reenactment.py"`
  - default pipeline backend: `FULL_BODY_REENACTMENT_PIPELINE_COMMAND="python3 /worker/src/full_body_reenactment_mimicmotion.py"`
  - `Dockerfile.generation` installs the official MimicMotion repo under `/opt/mimicmotion`
  - the worker exposes `/usr/local/bin/mimicmotion` as a wrapper binary
  - first request downloads MimicMotion weights into `/runpod-volume/truefaceswap-cache/mimicmotion/checkpoints` when a Runpod volume is attached
  - if no volume is attached, weights fall back to `/opt/mimicmotion/models/checkpoints`
  - this image sets `REQUIRE_FULL_BODY_REENACTMENT_BACKEND=true` by default
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
  - `LIVEPORTRAIT_REF` (Docker build arg, default `main`)
  - `LIVEPORTRAIT_HF_REPO`
  - `LIVEPORTRAIT_WEIGHTS_DIR`
  - `LIVEPORTRAIT_AUTO_CROP`
  - `LIVEPORTRAIT_AUDIO_PRIORITY`
  - `LIVEPORTRAIT_DRIVING_OPTION`
  - `LIVEPORTRAIT_ANIMATION_REGION`
  - `MIMICMOTION_REF` (Docker build arg, default `main`)
  - `MIMICMOTION_HF_REPO`
  - `MIMICMOTION_CKPT_NAME`
  - `MIMICMOTION_WEIGHTS_DIR`
  - `MIMICMOTION_BASE_MODEL`
  - `MIMICMOTION_SAMPLE_STRIDE`
  - `MIMICMOTION_NUM_INFERENCE_STEPS`
  - `MIMICMOTION_GUIDANCE_SCALE`
  - `MIMICMOTION_NOISE_AUG_STRENGTH`
  - `MIMICMOTION_FRAMES_OVERLAP`
  - `MIMICMOTION_RESOLUTION`
  - `GENERATION_MODEL_ID`
  - `GENERATION_MODEL_DTYPE=auto|bf16|fp16`
  - `GENERATION_OFFLOAD_MODE=model|sequential|none`
  - `GENERATION_MULTI_IMAGE_MODE=hero_grid|primary_only`
  - `GENERATION_HERO_IMAGE_INDEX`
  - `GENERATION_IDENTITY_VIDEO_MAX_FRAMES`
  - `GENERATION_IDENTITY_VIDEO_SAMPLE_FPS`
  - `GENERATION_IDENTITY_VIDEO_KEEP_FRAMES`
  - `GENERATION_MOTION_REFERENCE_MAX_FRAMES`
  - `GENERATION_MOTION_REFERENCE_SAMPLE_FPS`
  - `GENERATION_MOTION_CONDITIONING_MODE=prompt_only|direct_warp`
  - `GENERATION_MOTION_WARP_SCALE` (`direct_warp` experimental only)
  - `GENERATION_MOTION_WARP_STABILIZATION` (`direct_warp` experimental only)
  - `GENERATION_MOTION_WARP_SMOOTHING` (`direct_warp` experimental only)
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
  - Full-body backend is mandatory in this image: `mimicmotion`.
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
