#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video-driven full-body reenactment via HunyuanCustom")
    parser.add_argument("--shot-plan", required=True, dest="shot_plan")
    parser.add_argument("--identity-pack", required=True, dest="identity_pack")
    parser.add_argument("--control-bundle", required=True, dest="control_bundle")
    parser.add_argument("--source-image", required=True, dest="source_image")
    parser.add_argument("--driving-video", required=True, dest="driving_video")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []
    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.rstrip()
        lines.append(line)
        print(line, flush=True)
    return_code = process.wait()
    if return_code != 0:
        raise SystemExit(f"command failed: {' '.join(cmd)}\nstdout+stderr:\n" + "\n".join(lines))


def _find_generated_video(directory: Path) -> Path:
    candidates = sorted(
        (
            path
            for path in directory.rglob("*.mp4")
            if path.is_file() and path.stat().st_size > 0
        ),
        key=lambda path: (path.stat().st_mtime, path.stat().st_size),
        reverse=True,
    )
    if not candidates:
        raise SystemExit(f"HunyuanCustom produced no mp4 output under {directory}")
    return candidates[0]


def main() -> None:
    args = _parse_args()

    from generation_contract import ensure_video_output, load_control_bundle, load_shot_plan

    shot_plan = load_shot_plan(Path(args.shot_plan))
    control_bundle = load_control_bundle(Path(args.control_bundle))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mask_video = Path(control_bundle.mask_video) if control_bundle.mask_video else None
    if not mask_video or not mask_video.exists():
        raise SystemExit(
            "HunyuanCustom video-driven backend requires a driving mask video. "
            "Upload Driving Mask Video in the full-body form or provide one through the control bundle."
        )

    repo_dir = Path(os.getenv("HUNYUANCUSTOM_REPO_DIR", "/opt/hunyuancustom"))
    runner = os.getenv("HUNYUANCUSTOM_RUNNER", "python hymm_sp/sample_gpu_poor.py").strip()
    model_base = os.getenv("HUNYUANCUSTOM_MODEL_BASE", "").strip() or str(repo_dir / "models")
    checkpoint = os.getenv("HUNYUANCUSTOM_VIDEO_CKPT", "").strip() or str(
        Path(model_base) / "hunyuancustom_editing_720P" / "mp_rank_00_model_states.pt"
    )
    positive_prompt = (
        (shot_plan.prompt or "").strip()
        or os.getenv(
            "HUNYUANCUSTOM_DEFAULT_POS_PROMPT",
            "Realistic, High-quality. A person performs naturally with accurate body motion and consistent identity.",
        ).strip()
    )
    negative_prompt = (
        (shot_plan.negative_prompt or "").strip()
        or os.getenv(
            "HUNYUANCUSTOM_DEFAULT_NEG_PROMPT",
            "low quality, blurry, deformation, distorted face, bad hands, bad limbs, text, subtitles, static picture, black border",
        ).strip()
    )
    infer_steps = int(os.getenv("HUNYUANCUSTOM_INFER_STEPS", "50"))
    flow_shift = float(os.getenv("HUNYUANCUSTOM_FLOW_SHIFT_EVAL_VIDEO", "5.0"))
    expand_scale = int(os.getenv("HUNYUANCUSTOM_EXPAND_SCALE", "5"))
    pose_enhance = os.getenv("HUNYUANCUSTOM_POSE_ENHANCE", "true").strip().lower() in {"1", "true", "yes", "on"}
    seed = shot_plan.seed if shot_plan.seed is not None else int(os.getenv("HUNYUANCUSTOM_SEED", "1024"))

    if not repo_dir.exists():
        raise SystemExit(
            f"HunyuanCustom repo not found at {repo_dir}. Install the backend and set HUNYUANCUSTOM_REPO_DIR."
        )

    output_dir = output_path.parent / f"{output_path.stem}.hunyuancustom"
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("MODEL_BASE", model_base)
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{repo_dir}:{pythonpath}" if pythonpath else str(repo_dir)

    cmd = [
        *shlex.split(runner),
        "--ref-image",
        args.source_image,
        "--input-video",
        args.driving_video,
        "--mask-video",
        str(mask_video),
        "--expand-scale",
        str(expand_scale),
        "--video-condition",
        "--pos-prompt",
        positive_prompt,
        "--neg-prompt",
        negative_prompt,
        "--ckpt",
        checkpoint,
        "--seed",
        str(seed),
        "--infer-steps",
        str(infer_steps),
        "--flow-shift-eval-video",
        str(flow_shift),
        "--save-path",
        str(output_dir),
    ]
    if pose_enhance:
        cmd.append("--pose-enhance")

    _run(cmd, cwd=repo_dir, env=env)
    produced = _find_generated_video(output_dir)
    output_path.write_bytes(produced.read_bytes())
    ensure_video_output(output_path)


if __name__ == "__main__":
    main()
