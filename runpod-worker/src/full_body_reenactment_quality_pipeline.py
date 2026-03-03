#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Composed quality-first full-body reenactment pipeline")
    parser.add_argument("--shot-plan", required=True, dest="shot_plan")
    parser.add_argument("--identity-pack", required=True, dest="identity_pack")
    parser.add_argument("--control-bundle", required=True, dest="control_bundle")
    parser.add_argument("--source-image", required=True, dest="source_image")
    parser.add_argument("--driving-video", required=True, dest="driving_video")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    process = subprocess.Popen(
        cmd,
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


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def _select_base_command() -> str:
    strategy = (os.getenv("FULL_BODY_BASE_RENDER_STRATEGY", "mimicmotion").strip().lower() or "mimicmotion")
    if strategy in {"stronger", "quality", "next"}:
        return (
            os.getenv("FULL_BODY_STRONGER_RENDER_WRAPPER", "").strip()
            or "python3 /worker/src/full_body_reenactment_stronger_base.py"
        )
    return (
        os.getenv("FULL_BODY_BASE_RENDER_COMMAND", "").strip()
        or "python3 /worker/src/full_body_reenactment_mimicmotion.py"
    )


def main() -> None:
    args = _parse_args()

    from generation_contract import ensure_video_output, load_shot_plan

    shot_plan = load_shot_plan(Path(args.shot_plan))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_command = _select_base_command()
    face_refiner_command = (
        os.getenv("FULL_BODY_FACE_REFINER_COMMAND", "").strip()
        or "python3 /worker/src/full_body_face_refiner_gfpgan.py"
    )
    temporal_cleanup_command = (
        os.getenv("FULL_BODY_TEMPORAL_CLEANUP_COMMAND", "").strip()
        or "python3 /worker/src/full_body_temporal_cleanup_ffmpeg.py"
    )

    with tempfile.TemporaryDirectory(prefix="full-body-quality-pipeline-") as temp_dir:
        temp_root = Path(temp_dir)
        base_output = temp_root / "base.mp4"
        refined_output = temp_root / "refined.mp4"
        cleaned_output = temp_root / "cleaned.mp4"

        if "full_body_reenactment_quality_pipeline.py" in base_command:
            raise SystemExit("FULL_BODY_BASE_RENDER_COMMAND must point to a base renderer, not the quality pipeline itself")

        if Path(args.source_image).resolve() == output_path.resolve() or Path(args.driving_video).resolve() == output_path.resolve():
            raise SystemExit("invalid output target overlaps with input files")

        base_cmd = [
            *shlex.split(base_command),
            "--shot-plan",
            args.shot_plan,
            "--identity-pack",
            args.identity_pack,
            "--control-bundle",
            args.control_bundle,
            "--source-image",
            args.source_image,
            "--driving-video",
            args.driving_video,
            "--output",
            str(base_output),
        ]
        _run(base_cmd)
        ensure_video_output(base_output)

        current_input = base_output

        if face_refiner_command:
            face_cmd = [
                *shlex.split(face_refiner_command),
                "--shot-plan",
                args.shot_plan,
                "--identity-pack",
                args.identity_pack,
                "--control-bundle",
                args.control_bundle,
                "--source-image",
                args.source_image,
                "--driving-video",
                args.driving_video,
                "--input-video",
                str(current_input),
                "--output",
                str(refined_output),
            ]
            _run(face_cmd)
            ensure_video_output(refined_output)
            current_input = refined_output
        else:
            _copy(current_input, refined_output)
            current_input = refined_output

        if temporal_cleanup_command:
            cleanup_cmd = [
                *shlex.split(temporal_cleanup_command),
                "--input-video",
                str(current_input),
                "--output",
                str(cleaned_output),
                "--quality",
                shot_plan.render_profile.quality,
            ]
            _run(cleanup_cmd)
            ensure_video_output(cleaned_output)
            current_input = cleaned_output

        _copy(current_input, output_path)


if __name__ == "__main__":
    main()
