# Generation Runner Contract

The generation worker uses two JSON contracts and two optional adapter commands.

Current first real backend in this repo:

- portrait reenactment wrapper: `/worker/src/generation_render_reenactment.py`
- full body reenactment wrapper: `/worker/src/generation_render_full_body_reenactment.py`
- portrait reenactment backend: `/worker/src/portrait_reenactment_liveportrait.py`
- full body reenactment backend: `/worker/src/full_body_reenactment_mimicmotion.py`
- render: `/worker/src/generation_render_cogvideox.py`
- refine: `/worker/src/generation_refine_basic.py`

## Files

- `identity_pack.json`
  - produced by worker before generation
  - contains identity image inventory and optional identity video
- `control_bundle.json`
  - produced by worker before render for driving-video reenactment
  - contains sampled driving frames and motion metadata
- `shot_plan.json`
  - produced by worker before render
  - contains task type, prompt, motion/style presets, duration, seed, render profile, and optional control bundle path

Contract parsing and validation lives in:

- `/worker/src/generation_contract.py`

## Required render adapter CLI

```bash
<render-command> --shot-plan /path/to/shot_plan.json --output /path/to/rendered.mp4
```

Portrait reenactment render adapters may also receive:

```bash
--identity-pack /path/to/identity_pack.json --control-bundle /path/to/control_bundle.json
```

Optional:

```bash
--report /path/to/render-report.json
```

## Required refine adapter CLI

```bash
<refine-command> --identity-pack /path/to/identity_pack.json --input /path/to/rendered.mp4 --output /path/to/final.mp4
```

Optional:

```bash
--report /path/to/refine-report.json
```

## Report format

If provided, report JSON must match `AdapterReport` in `generation_contract.py`:

- `version`
- `stage`
- `engine`
- `model`
- `metrics`
- `warnings`

## Worker behavior

- If adapter command supports `--report`, worker validates the report.
- If adapter command rejects `--report`, worker retries once without it for backward compatibility.
- Worker always validates that output video exists and is non-empty.

## Example adapters

The generation image ships with demo adapters:

- `/worker/scripts/example_generation_render.py`
- `/worker/scripts/example_generation_refine.py`

They are placeholders for local verification only.

## First real backend limitations

The initial self-hosted backend is intentionally narrow:

- supports `portrait_reenactment` as a first-class worker task
- supports `full_body_reenactment` as a separate first-class worker task with its own render wrapper
- routes `portrait_reenactment` through a dedicated wrapper backend instead of the generic CogVideoX adapter
- routes `full_body_reenactment` through its own wrapper and defaults to a dedicated MimicMotion-based backend command
- includes a concrete LivePortrait-based reenactment backend entrypoint in-repo
- includes a concrete MimicMotion-based full-body reenactment backend entrypoint in-repo
- `Dockerfile.generation` is intentionally slimmed for the production full-body path and does not bundle the portrait runtime
- `Dockerfile.generation` installs the official MimicMotion runtime into `/opt/mimicmotion` and exposes `/usr/local/bin/mimicmotion`
- MimicMotion weights are downloaded lazily on first use and should be cached on the mounted Runpod volume when available
- uses CogVideoX image-to-video on Runpod GPU
- can convert up to 4 identity images into one reference canvas for the model input
- can be forced back to one image with `GENERATION_MULTI_IMAGE_MODE=primary_only`
- does not yet do learned multi-image identity fusion
- can sample identity-video frames and add the strongest ones into the identity pack
- can extract a control bundle from a driving video so reenactment assets remain first-class inputs
- can analyze motion-reference video and convert it into a motion profile that is appended to prompting
- motion-reference video is currently reduced to motion-profile prompting in the CogVideoX backend
- `GENERATION_MOTION_CONDITIONING_MODE=direct_warp` exists only as an experimental post-generation fallback, not as true model-level motion tracking
- refine stage is currently a passthrough copy stage

This is still a real self-hosted generation path, but not the final industry-grade identity stack yet.
