#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Example local generation render adapter")
    parser.add_argument("--shot-plan", required=True, dest="shot_plan")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report")
    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(cmd)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")


def main() -> None:
    args = _parse_args()

    from generation_contract import (  # imported late so script works with PYTHONPATH=/worker/src
        CONTRACT_VERSION,
        AdapterReport,
        load_identity_pack,
        load_shot_plan,
        save_adapter_report,
    )

    shot_plan = load_shot_plan(Path(args.shot_plan))
    identity_pack = load_identity_pack(Path(shot_plan.identity_pack_path))
    if not identity_pack.primary_image:
        raise SystemExit("identity pack primary_image is required")

    resolution = shot_plan.render_profile.resolution
    fps = shot_plan.render_profile.fps
    duration = shot_plan.duration_seconds
    size = f"{resolution[0]}:{resolution[1]}"
    zoom_size = f"{resolution[0]}x{resolution[1]}"
    zoom_speed = "0.0018" if shot_plan.render_profile.quality == "max" else "0.0012"
    vf = (
        f"scale={size}:force_original_aspect_ratio=increase,"
        f"crop={size},"
        f"zoompan=z='min(zoom+{zoom_speed},1.18)':d=1:"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={zoom_size}:fps={fps},"
        "format=yuv420p"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-loop",
        "1",
        "-i",
        identity_pack.primary_image,
        "-vf",
        vf,
        "-t",
        str(duration),
        "-r",
        str(fps),
        "-c:v",
        "libx264",
        args.output,
    ]
    _run(cmd)

    if args.report:
        save_adapter_report(
            Path(args.report),
            AdapterReport(
                version=CONTRACT_VERSION,
                stage="generating",
                engine="example-local-render",
                model="ffmpeg-zoompan-demo",
                metrics={
                    "duration_seconds": duration,
                    "fps": fps,
                    "image_count": len(identity_pack.images),
                },
                warnings=["Example adapter only. Replace with your real local generation model."],
            ),
        )


if __name__ == "__main__":
    main()
