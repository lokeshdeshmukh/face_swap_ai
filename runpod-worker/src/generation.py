from __future__ import annotations

import hashlib
import logging
import os
import shlex
import subprocess
from pathlib import Path
from typing import Callable, List, Optional

from generation_contract import (
    CONTRACT_VERSION,
    AdapterReport,
    ContractError,
    IdentityImage,
    IdentityPack,
    RenderProfile,
    ShotPlan,
    ensure_video_output,
    load_adapter_report,
    load_identity_pack,
    load_shot_plan,
    save_adapter_report,
    save_identity_pack,
    save_shot_plan,
)

logger = logging.getLogger("runpod-worker")


class GenerationError(RuntimeError):
    pass


ProgressCallback = Callable[[str, dict[str, object] | None], None]


def _run(cmd: List[str]) -> str:
    logger.info("running command: %s", " ".join(cmd))
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
        logger.info("[generation-cmd] %s", line)
    return_code = process.wait()
    output = "\n".join(lines)
    if return_code != 0:
        raise GenerationError(f"command failed: {' '.join(cmd)}\nstdout+stderr:\n{output}")
    return output


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_identity_pack(
    source_images: List[Path],
    identity_video: Optional[Path],
    output_dir: Path,
    progress: Optional[ProgressCallback] = None,
) -> Path:
    if progress:
        progress("identity_pack", {"image_count": len(source_images), "has_identity_video": bool(identity_video)})

    output_dir.mkdir(parents=True, exist_ok=True)
    pack_path = output_dir / "identity_pack.json"
    pack = IdentityPack(
        version=CONTRACT_VERSION,
        primary_image=str(source_images[0]) if source_images else None,
        images=[
            IdentityImage(
                path=str(path),
                name=path.name,
                sha256=_sha256_file(path),
            )
            for path in source_images
        ],
        identity_video=str(identity_video) if identity_video and identity_video.exists() else None,
    )
    try:
        save_identity_pack(pack_path, pack)
    except ContractError as exc:
        raise GenerationError(str(exc)) from exc
    return pack_path


def build_shot_plan(
    identity_pack_path: Path,
    motion_reference_video: Optional[Path],
    driving_audio: Optional[Path],
    output_dir: Path,
    quality: str,
    aspect_ratio: str,
    job_config: dict[str, object],
    progress: Optional[ProgressCallback] = None,
) -> Path:
    if progress:
        progress(
            "shot_plan",
            {
                "motion_preset": job_config.get("motion_preset"),
                "style_preset": job_config.get("style_preset"),
                "duration_seconds": job_config.get("duration_seconds"),
            },
        )

    shot_plan_path = output_dir / "shot_plan.json"
    fps = 24 if quality == "max" else 20
    plan = ShotPlan(
        version=CONTRACT_VERSION,
        identity_pack_path=str(identity_pack_path),
        prompt=str(job_config.get("prompt") or ""),
        negative_prompt=str(job_config["negative_prompt"]) if job_config.get("negative_prompt") else None,
        motion_preset=str(job_config["motion_preset"]) if job_config.get("motion_preset") else None,
        style_preset=str(job_config["style_preset"]) if job_config.get("style_preset") else None,
        duration_seconds=int(job_config.get("duration_seconds", 5)),
        seed=int(job_config["seed"]) if job_config.get("seed") is not None else None,
        motion_reference_video=(
            str(motion_reference_video) if motion_reference_video and motion_reference_video.exists() else None
        ),
        driving_audio=str(driving_audio) if driving_audio and driving_audio.exists() else None,
        render_profile=RenderProfile(
            quality=quality,
            aspect_ratio=aspect_ratio,
            fps=fps,
            resolution={
                "9:16": [1080, 1920],
                "1:1": [1080, 1080],
                "4:5": [1080, 1350],
            }.get(aspect_ratio, [1080, 1920]),
            frame_count=int(job_config.get("duration_seconds", 5)) * fps,
        ),
    )
    try:
        save_shot_plan(shot_plan_path, plan)
    except ContractError as exc:
        raise GenerationError(str(exc)) from exc
    return shot_plan_path


def _run_placeholder_preview(shot_plan_path: Path, output_video: Path) -> None:
    shot_plan = load_shot_plan(shot_plan_path)
    identity_pack = load_identity_pack(Path(shot_plan.identity_pack_path))
    primary_image = identity_pack.primary_image
    if not primary_image:
        raise GenerationError("placeholder generation requires a primary identity image")
    resolution = shot_plan.render_profile.resolution
    fps = shot_plan.render_profile.fps
    duration = shot_plan.duration_seconds or 5
    zoom_speed = "0.0015" if shot_plan.render_profile.quality == "max" else "0.0010"
    size = f"{resolution[0]}:{resolution[1]}"
    zoom_size = f"{resolution[0]}x{resolution[1]}"
    vf = (
        f"scale={size}:force_original_aspect_ratio=increase,"
        f"crop={size},"
        f"zoompan=z='min(zoom+{zoom_speed},1.15)':d=1:"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={zoom_size}:fps={fps},"
        "format=yuv420p"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-loop",
        "1",
        "-i",
        str(primary_image),
        "-vf",
        vf,
        "-t",
        str(duration),
        "-r",
        str(fps),
        "-c:v",
        "libx264",
        str(output_video),
    ]
    _run(cmd)
    ensure_video_output(output_video)


def render_generation(
    shot_plan_path: Path,
    output_video: Path,
    progress: Optional[ProgressCallback] = None,
) -> None:
    if progress:
        progress("generating", None)

    render_command = os.getenv("GENERATION_RENDER_COMMAND", "").strip() or os.getenv(
        "GENERATION_PIPELINE_COMMAND", ""
    ).strip()
    report_path = output_video.with_suffix(".render-report.json")
    if render_command:
        cmd = [
            *shlex.split(render_command),
            "--shot-plan",
            str(shot_plan_path),
            "--output",
            str(output_video),
            "--report",
            str(report_path),
        ]
        try:
            _run(cmd)
        except GenerationError as exc:
            if "--report" in str(exc):
                fallback_cmd = [*shlex.split(render_command), "--shot-plan", str(shot_plan_path), "--output", str(output_video)]
                _run(fallback_cmd)
            else:
                raise
        ensure_video_output(output_video)
        if report_path.exists():
            try:
                load_adapter_report(report_path)
            except ContractError as exc:
                raise GenerationError(f"invalid render adapter report: {exc}") from exc
        return

    _run_placeholder_preview(shot_plan_path, output_video)
    save_adapter_report(
        report_path,
        AdapterReport(
            version=CONTRACT_VERSION,
            stage="generating",
            engine="placeholder",
            model="ffmpeg-zoompan",
            metrics={"mode": "placeholder"},
            warnings=["GENERATION_RENDER_COMMAND not set; produced placeholder preview"],
        ),
    )
    logger.warning("GENERATION_RENDER_COMMAND not set; produced placeholder preview instead of real generation")


def refine_generation(
    rendered_video: Path,
    identity_pack_path: Path,
    output_video: Path,
    progress: Optional[ProgressCallback] = None,
) -> None:
    if progress:
        progress("identity_refine", None)

    refine_command = os.getenv("GENERATION_REFINE_COMMAND", "").strip()
    report_path = output_video.with_suffix(".refine-report.json")
    if refine_command:
        cmd = [
            *shlex.split(refine_command),
            "--identity-pack",
            str(identity_pack_path),
            "--input",
            str(rendered_video),
            "--output",
            str(output_video),
            "--report",
            str(report_path),
        ]
        try:
            _run(cmd)
        except GenerationError as exc:
            if "--report" in str(exc):
                fallback_cmd = [
                    *shlex.split(refine_command),
                    "--identity-pack",
                    str(identity_pack_path),
                    "--input",
                    str(rendered_video),
                    "--output",
                    str(output_video),
                ]
                _run(fallback_cmd)
            else:
                raise
        ensure_video_output(output_video)
        if report_path.exists():
            try:
                load_adapter_report(report_path)
            except ContractError as exc:
                raise GenerationError(f"invalid refine adapter report: {exc}") from exc
        return

    if rendered_video != output_video:
        output_video.write_bytes(rendered_video.read_bytes())
    ensure_video_output(output_video)
    save_adapter_report(
        report_path,
        AdapterReport(
            version=CONTRACT_VERSION,
            stage="identity_refine",
            engine="passthrough",
            model="copy",
            metrics={"mode": "passthrough"},
            warnings=["GENERATION_REFINE_COMMAND not set; returned rendered output unchanged"],
        ),
    )
