#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wrapper for a stronger full-body base renderer")
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


def main() -> None:
    args = _parse_args()
    backend_command = os.getenv("FULL_BODY_STRONGER_RENDER_COMMAND", "").strip()
    backend_name = os.getenv("FULL_BODY_STRONGER_RENDER_BACKEND", "unconfigured").strip() or "unconfigured"
    if not backend_command:
        raise SystemExit(
            "stronger full-body base renderer is not configured; set FULL_BODY_STRONGER_RENDER_COMMAND "
            f"for backend '{backend_name}'"
        )
    cmd = [
        *shlex.split(backend_command),
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
        args.output,
    ]
    _run(cmd)
    output = Path(args.output)
    if not output.exists() or output.stat().st_size <= 0:
        raise SystemExit(f"stronger base renderer '{backend_name}' produced no output video")


if __name__ == "__main__":
    main()
