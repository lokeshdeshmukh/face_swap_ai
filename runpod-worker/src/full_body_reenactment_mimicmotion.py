#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-body reenactment backend using MimicMotion")
    parser.add_argument("--shot-plan", required=True, dest="shot_plan")
    parser.add_argument("--identity-pack", required=True, dest="identity_pack")
    parser.add_argument("--control-bundle", required=True, dest="control_bundle")
    parser.add_argument("--source-image", required=True, dest="source_image")
    parser.add_argument("--driving-video", required=True, dest="driving_video")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(cmd)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")


def _has_audio_stream(video_path: Path) -> bool:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False
    return "audio" in result.stdout.lower()


def _restore_audio_if_missing(audio_source: Path, output_video: Path) -> None:
    if not audio_source.exists() or not output_video.exists():
        return
    if not _has_audio_stream(audio_source) or _has_audio_stream(output_video):
        return

    remuxed = output_video.with_name(f"{output_video.stem}.with-audio{output_video.suffix}")
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(output_video),
            "-i",
            str(audio_source),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(remuxed),
        ]
    )
    remuxed.replace(output_video)


def main() -> None:
    args = _parse_args()

    from generation_contract import load_control_bundle, load_identity_pack, load_shot_plan

    if shutil.which("mimicmotion") is None:
        raise SystemExit("mimicmotion executable not found in worker image")

    shot_plan = load_shot_plan(Path(args.shot_plan))
    identity_pack = load_identity_pack(Path(args.identity_pack))
    control_bundle = load_control_bundle(Path(args.control_bundle))

    if shot_plan.task_type != "full_body_reenactment":
        raise SystemExit(
            f"full_body_reenactment backend expected task_type=full_body_reenactment, got {shot_plan.task_type}"
        )

    source_image = Path(args.source_image)
    driving_video = Path(args.driving_video)
    output_path = Path(args.output)
    if not source_image.exists():
        raise SystemExit(f"source image missing: {source_image}")
    if not driving_video.exists():
        raise SystemExit(f"driving video missing: {driving_video}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    render_profile = shot_plan.render_profile
    # MimicMotion 1.1 is materially less stable when fed UI/render-profile sizes directly.
    # Keep the runtime inside the model's practical operating range and expose overrides via env.
    resolution = int(os.getenv("MIMICMOTION_RESOLUTION", "576"))
    resolution = max(512, min(resolution, 1024))
    max_frames = int(os.getenv("MIMICMOTION_MAX_FRAMES", "72"))
    min_frames = int(os.getenv("MIMICMOTION_MIN_FRAMES", "16"))
    frame_count = min(render_profile.frame_count, max_frames)
    frame_count = max(frame_count, min_frames)
    fps = min(render_profile.fps, int(os.getenv("MIMICMOTION_MAX_FPS", "15")))
    fps = max(fps, 1)
    with tempfile.TemporaryDirectory(prefix="mimicmotion-full-body-") as temp_dir:
        temp_path = Path(temp_dir)
        rendered = temp_path / "rendered.mp4"
        cmd = [
            "mimicmotion",
            "--source-image",
            str(source_image),
            "--driving-video",
            str(driving_video),
            "--output",
            str(rendered),
            "--frame-count",
            str(frame_count),
            "--fps",
            str(fps),
            "--resolution",
            str(resolution),
        ]
        if shot_plan.seed is not None:
            cmd.extend(["--seed", str(shot_plan.seed)])
        _run(cmd)

        if not rendered.exists() or rendered.stat().st_size <= 0:
            raise SystemExit("mimicmotion produced no output video")

        shutil.copyfile(rendered, output_path)

    if shot_plan.driving_audio:
        _restore_audio_if_missing(Path(shot_plan.driving_audio), output_path)
    else:
        _restore_audio_if_missing(Path(control_bundle.driving_video), output_path)


if __name__ == "__main__":
    main()
