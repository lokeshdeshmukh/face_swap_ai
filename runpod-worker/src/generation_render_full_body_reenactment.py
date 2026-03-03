#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render full-body reenactment from identity pack and control bundle")
    parser.add_argument("--shot-plan", required=True, dest="shot_plan")
    parser.add_argument("--identity-pack", dest="identity_pack")
    parser.add_argument("--control-bundle", dest="control_bundle")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report")
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

    from generation_contract import (
        CONTRACT_VERSION,
        AdapterReport,
        ensure_video_output,
        load_control_bundle,
        load_identity_pack,
        load_shot_plan,
        save_adapter_report,
    )

    shot_plan = load_shot_plan(Path(args.shot_plan))
    if shot_plan.task_type != "full_body_reenactment":
        raise SystemExit(
            f"generation_render_full_body_reenactment only supports full_body_reenactment, got {shot_plan.task_type}"
        )

    identity_pack_path = Path(args.identity_pack or shot_plan.identity_pack_path)
    identity_pack = load_identity_pack(identity_pack_path)

    control_bundle_path = Path(args.control_bundle or shot_plan.control_bundle_path or "")
    if not control_bundle_path.exists():
        raise SystemExit("full body reenactment requires a control bundle")
    control_bundle = load_control_bundle(control_bundle_path)

    primary_image = identity_pack.primary_image or (identity_pack.images[0].path if identity_pack.images else None)
    if not primary_image:
        raise SystemExit("full body reenactment requires at least one identity image")

    backend_command = os.getenv("FULL_BODY_REENACTMENT_PIPELINE_COMMAND", "").strip()
    backend_name = os.getenv("FULL_BODY_REENACTMENT_BACKEND", "").strip() or "unconfigured"
    if not backend_command:
        raise SystemExit(
            "full body reenactment backend is not configured; set FULL_BODY_REENACTMENT_PIPELINE_COMMAND "
            "to an in-repo full-body motion transfer runner"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        *shlex.split(backend_command),
        "--shot-plan",
        str(args.shot_plan),
        "--identity-pack",
        str(identity_pack_path),
        "--control-bundle",
        str(control_bundle_path),
        "--source-image",
        str(primary_image),
        "--driving-video",
        str(control_bundle.driving_video),
        "--output",
        str(output_path),
    ]
    _run(cmd)
    ensure_video_output(output_path)

    if args.report:
        warnings = [
            "Full-body reenactment quality depends on the configured render pipeline; MimicMotion is only the default base renderer, not the final quality ceiling."
        ]
        if len(identity_pack.images) > 1:
            warnings.append("Multiple identity images were passed into the full-body reenactment wrapper.")
        if identity_pack.identity_video:
            warnings.append("Identity video support depends on the configured full-body reenactment backend.")
        save_adapter_report(
            Path(args.report),
            AdapterReport(
                version=CONTRACT_VERSION,
                stage="generating",
                engine="full_body_reenactment_wrapper",
                model=backend_name,
                metrics={
                    "task_type": shot_plan.task_type,
                    "identity_images": len(identity_pack.images),
                    "control_frames": control_bundle.sampled_frames,
                    "sample_fps": control_bundle.sample_fps,
                    "motion_type": control_bundle.motion_type or "unknown",
                },
                warnings=warnings,
            ),
        )


if __name__ == "__main__":
    main()
