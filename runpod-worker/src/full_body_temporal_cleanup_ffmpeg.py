#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Temporal cleanup / sharpen pass for generated reenactment video")
    parser.add_argument("--input-video", required=True, dest="input_video")
    parser.add_argument("--output", required=True)
    parser.add_argument("--quality", default="balanced")
    return parser.parse_args()


def _filter_for_quality(quality: str) -> tuple[str, str]:
    quality_key = (quality or "balanced").strip().lower()
    if quality_key == "max":
        return ("hqdn3d=1.2:1.2:5.0:5.0,unsharp=5:5:0.9:5:5:0.0", "17")
    if quality_key == "fast":
        return ("hqdn3d=0.8:0.8:3.5:3.5,unsharp=3:3:0.4:3:3:0.0", "21")
    return ("hqdn3d=1.0:1.0:4.5:4.5,unsharp=5:5:0.7:5:5:0.0", "19")


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        output = (result.stdout or "") + ("\n" if result.stdout and result.stderr else "") + (result.stderr or "")
        raise SystemExit(f"command failed: {' '.join(cmd)}\nstdout+stderr:\n{output.strip()}")


def main() -> None:
    args = _parse_args()
    input_video = Path(args.input_video)
    output_video = Path(args.output)
    if not input_video.exists():
        raise SystemExit(f"input video missing: {input_video}")
    output_video.parent.mkdir(parents=True, exist_ok=True)

    vf, crf = _filter_for_quality(args.quality)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_video),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        crf,
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(output_video),
    ]
    _run(cmd)


if __name__ == "__main__":
    main()
