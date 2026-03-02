# Generation Runner Contract

The generation worker uses two JSON contracts and two optional adapter commands.

## Files

- `identity_pack.json`
  - produced by worker before generation
  - contains identity image inventory and optional identity video
- `shot_plan.json`
  - produced by worker before render
  - contains prompt, motion/style presets, duration, seed, and render profile

Contract parsing and validation lives in:

- `/worker/src/generation_contract.py`

## Required render adapter CLI

```bash
<render-command> --shot-plan /path/to/shot_plan.json --output /path/to/rendered.mp4
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
