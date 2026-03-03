from __future__ import annotations

import hashlib
import logging
import math
import os
import shlex
import subprocess
from pathlib import Path
from typing import Callable, List, Optional

from generation_contract import (
    CONTRACT_VERSION,
    AdapterReport,
    ControlBundle,
    ContractError,
    IdentityImage,
    IdentityPack,
    RenderProfile,
    ShotPlan,
    ensure_video_output,
    load_adapter_report,
    load_control_bundle,
    load_identity_pack,
    load_shot_plan,
    save_adapter_report,
    save_control_bundle,
    save_identity_pack,
    save_shot_plan,
)

logger = logging.getLogger("runpod-worker")


class GenerationError(RuntimeError):
    pass


ProgressCallback = Callable[[str, dict[str, object] | None], None]


def _report_flag_unsupported(error_text: str) -> bool:
    lowered = error_text.lower()
    return (
        "unrecognized arguments: --report" in lowered
        or "unknown option --report" in lowered
        or "unknown argument --report" in lowered
        or "no such option: --report" in lowered
    )


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


def _extract_identity_video_frames(
    identity_video: Path,
    output_dir: Path,
    max_frames: int,
    fps: float,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / "identity_%03d.jpg"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(identity_video),
        "-vf",
        f"fps={fps}",
        "-frames:v",
        str(max_frames),
        str(pattern),
    ]
    _run(cmd)
    return sorted(output_dir.glob("identity_*.jpg"))


def _score_identity_frame(path: Path) -> float:
    from PIL import Image, ImageFilter, ImageStat

    image = Image.open(path).convert("RGB")
    grayscale = image.convert("L")
    stat = ImageStat.Stat(grayscale)
    contrast = (stat.stddev[0] / 64.0) if stat.stddev else 0.0

    edges = grayscale.filter(ImageFilter.FIND_EDGES)
    edge_stat = ImageStat.Stat(edges)
    edge_mean = (edge_stat.mean[0] / 255.0) if edge_stat.mean else 0.0

    width, height = image.size
    megapixels = min((width * height) / 1_000_000.0, 12.0)
    aspect_ratio = width / max(height, 1)
    portrait_target = 0.75
    aspect_score = max(0.0, 1.0 - abs(aspect_ratio - portrait_target))

    return (megapixels * 0.35) + (contrast * 1.5) + (edge_mean * 4.0) + aspect_score


def _probe_video_metadata(video_path: Path) -> dict[str, float]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=0",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise GenerationError(f"ffprobe failed for {video_path}: {result.stderr}")

    duration = 0.0
    fps = 0.0
    for line in result.stdout.splitlines():
        if line.startswith("duration="):
            try:
                duration = float(line.split("=", 1)[1].strip())
            except ValueError:
                duration = 0.0
        elif line.startswith("avg_frame_rate="):
            rate = line.split("=", 1)[1].strip()
            if "/" in rate:
                num, den = rate.split("/", 1)
                try:
                    den_value = float(den)
                    fps = float(num) / den_value if den_value else 0.0
                except ValueError:
                    fps = 0.0
    return {"duration_seconds": duration, "source_fps": fps}


def _extract_motion_reference_frames(
    motion_reference_video: Path,
    output_dir: Path,
    max_frames: int,
    fps: float,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / "motion_%03d.jpg"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(motion_reference_video),
        "-vf",
        f"fps={fps}",
        "-frames:v",
        str(max_frames),
        str(pattern),
    ]
    _run(cmd)
    return sorted(output_dir.glob("motion_*.jpg"))


def analyze_motion_reference_video(motion_reference_video: Optional[Path], output_dir: Path) -> dict[str, object] | None:
    if not motion_reference_video or not motion_reference_video.exists():
        return None

    import cv2

    metadata = _probe_video_metadata(motion_reference_video)
    max_frames = int(os.getenv("GENERATION_MOTION_REFERENCE_MAX_FRAMES", "16"))
    sample_fps = float(os.getenv("GENERATION_MOTION_REFERENCE_SAMPLE_FPS", "2.0"))
    extracted = _extract_motion_reference_frames(
        motion_reference_video,
        output_dir / "motion_reference_frames",
        max_frames=max_frames,
        fps=sample_fps,
    )
    if len(extracted) < 2:
        return {
            **metadata,
            "sampled_frame_count": len(extracted),
            "motion_type": "insufficient_frames",
            "motion_summary": "reference clip too short for motion analysis",
        }

    tx_values: list[float] = []
    ty_values: list[float] = []
    scale_values: list[float] = []
    rotation_values: list[float] = []
    usable_pairs = 0
    frame_width = 0
    frame_height = 0

    previous = cv2.imread(str(extracted[0]), cv2.IMREAD_GRAYSCALE)
    if previous is None:
        raise GenerationError("failed to read extracted motion reference frame")
    frame_height, frame_width = previous.shape[:2]

    for current_path in extracted[1:]:
        current = cv2.imread(str(current_path), cv2.IMREAD_GRAYSCALE)
        if current is None:
            continue
        points = cv2.goodFeaturesToTrack(previous, maxCorners=200, qualityLevel=0.01, minDistance=7)
        if points is None or len(points) < 8:
            previous = current
            continue
        tracked, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None)
        if tracked is None or status is None:
            previous = current
            continue
        src = points[status.flatten() == 1]
        dst = tracked[status.flatten() == 1]
        if len(src) < 8 or len(dst) < 8:
            previous = current
            continue
        matrix, _ = cv2.estimateAffinePartial2D(src, dst)
        if matrix is None:
            previous = current
            continue

        a = float(matrix[0, 0])
        b = float(matrix[0, 1])
        tx = float(matrix[0, 2])
        ty = float(matrix[1, 2])
        scale = math.sqrt((a * a) + (b * b))
        rotation = math.degrees(math.atan2(b, a))

        tx_values.append(tx / max(frame_width, 1))
        ty_values.append(ty / max(frame_height, 1))
        scale_values.append(scale)
        rotation_values.append(rotation)
        usable_pairs += 1
        previous = current

    if not usable_pairs:
        return {
            **metadata,
            "sampled_frame_count": len(extracted),
            "motion_type": "unresolved",
            "motion_summary": "motion analysis could not resolve a stable transform",
        }

    mean_tx = sum(tx_values) / usable_pairs
    mean_ty = sum(ty_values) / usable_pairs
    mean_scale = sum(scale_values) / usable_pairs
    mean_rotation = sum(rotation_values) / usable_pairs
    net_translation = math.sqrt((sum(tx_values) ** 2) + (sum(ty_values) ** 2))
    motion_intensity = sum(abs(v) for v in tx_values + ty_values) / (usable_pairs * 2)
    jitter = (
        sum(abs(value - mean_tx) for value in tx_values) + sum(abs(value - mean_ty) for value in ty_values)
    ) / (usable_pairs * 2)

    motion_type = "locked_portrait"
    motion_summary = "mostly locked portrait framing"
    if abs(mean_scale - 1.0) >= 0.015:
        if mean_scale > 1.0:
            motion_type = "push_in"
            motion_summary = "camera gradually pushes in"
        else:
            motion_type = "pull_out"
            motion_summary = "camera gradually pulls back"
    elif motion_intensity <= 0.004:
        motion_type = "locked_portrait"
        motion_summary = "mostly locked portrait framing"
    elif jitter > max(net_translation, 0.01) * 0.8:
        motion_type = "handheld"
        motion_summary = "handheld camera movement with natural jitter"
    elif abs(sum(tx_values)) > abs(sum(ty_values)) * 1.25:
        if sum(tx_values) > 0:
            motion_type = "pan_right"
            motion_summary = "camera drifts toward the right"
        else:
            motion_type = "pan_left"
            motion_summary = "camera drifts toward the left"
    elif abs(sum(ty_values)) > abs(sum(tx_values)) * 1.25:
        if sum(ty_values) > 0:
            motion_type = "tilt_down"
            motion_summary = "camera tilts downward"
        else:
            motion_type = "tilt_up"
            motion_summary = "camera tilts upward"
    else:
        motion_type = "mixed_motion"
        motion_summary = "camera has mixed directional movement"

    return {
        **metadata,
        "sampled_frame_count": len(extracted),
        "usable_pairs": usable_pairs,
        "motion_type": motion_type,
        "motion_summary": motion_summary,
        "mean_translation_x": round(mean_tx, 5),
        "mean_translation_y": round(mean_ty, 5),
        "mean_scale": round(mean_scale, 5),
        "mean_rotation_degrees": round(mean_rotation, 3),
        "motion_intensity": round(motion_intensity, 5),
        "motion_jitter": round(jitter, 5),
    }


def build_control_bundle(
    driving_video: Optional[Path],
    mask_video: Optional[Path],
    output_dir: Path,
    progress: Optional[ProgressCallback] = None,
) -> Path | None:
    if not driving_video or not driving_video.exists():
        return None

    metadata = _probe_video_metadata(driving_video)
    max_frames = int(os.getenv("GENERATION_MOTION_REFERENCE_MAX_FRAMES", "16"))
    sample_fps = float(os.getenv("GENERATION_MOTION_REFERENCE_SAMPLE_FPS", "2.0"))
    frame_dir = output_dir / "control_frames"
    extracted = _extract_motion_reference_frames(
        driving_video,
        frame_dir,
        max_frames=max_frames,
        fps=sample_fps,
    )
    if not extracted:
        raise GenerationError("no control frames could be extracted from driving video")

    motion_profile = analyze_motion_reference_video(driving_video, output_dir)
    if progress:
        progress(
            "control_extract",
            {
                "sampled_frames": len(extracted),
                "sample_fps": sample_fps,
                "motion_type": motion_profile.get("motion_type") if motion_profile else None,
            },
        )

    bundle_path = output_dir / "control_bundle.json"
    bundle = ControlBundle(
        version=CONTRACT_VERSION,
        driving_video=str(driving_video),
        mask_video=str(mask_video) if mask_video and mask_video.exists() else None,
        frame_dir=str(frame_dir),
        sampled_frames=len(extracted),
        sample_fps=sample_fps,
        duration_seconds=metadata.get("duration_seconds"),
        source_fps=metadata.get("source_fps"),
        motion_type=str(motion_profile.get("motion_type")) if motion_profile and motion_profile.get("motion_type") else None,
        motion_summary=(
            str(motion_profile.get("motion_summary"))
            if motion_profile and motion_profile.get("motion_summary")
            else None
        ),
    )
    try:
        save_control_bundle(bundle_path, bundle)
    except ContractError as exc:
        raise GenerationError(str(exc)) from exc
    return bundle_path


def augment_identity_images(
    source_images: List[Path],
    identity_video: Optional[Path],
    output_dir: Path,
) -> tuple[list[Path], dict[str, object]]:
    combined_images = list(source_images)
    metadata: dict[str, object] = {
        "input_image_count": len(source_images),
        "identity_video_frame_count": 0,
        "identity_video_selected_count": 0,
    }

    if not identity_video or not identity_video.exists():
        return combined_images, metadata

    max_frames = int(os.getenv("GENERATION_IDENTITY_VIDEO_MAX_FRAMES", "12"))
    sample_fps = float(os.getenv("GENERATION_IDENTITY_VIDEO_SAMPLE_FPS", "1.5"))
    keep_frames = int(os.getenv("GENERATION_IDENTITY_VIDEO_KEEP_FRAMES", "3"))

    extracted = _extract_identity_video_frames(identity_video, output_dir / "identity_video_frames", max_frames, sample_fps)
    metadata["identity_video_frame_count"] = len(extracted)
    if not extracted:
        return combined_images, metadata

    ranked = sorted(extracted, key=_score_identity_frame, reverse=True)
    selected = ranked[: max(1, keep_frames)]
    combined_images.extend(selected)
    metadata["identity_video_selected_count"] = len(selected)
    metadata["identity_video_mode"] = "ranked_frames"
    return combined_images, metadata


def build_identity_pack(
    source_images: List[Path],
    identity_video: Optional[Path],
    output_dir: Path,
    progress: Optional[ProgressCallback] = None,
) -> Path:
    combined_images, augmentation_metadata = augment_identity_images(source_images, identity_video, output_dir)
    if progress:
        progress(
            "identity_pack",
            {
                "image_count": len(combined_images),
                "has_identity_video": bool(identity_video),
                **augmentation_metadata,
            },
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    pack_path = output_dir / "identity_pack.json"
    pack = IdentityPack(
        version=CONTRACT_VERSION,
        primary_image=str(combined_images[0]) if combined_images else None,
        images=[
            IdentityImage(
                path=str(path),
                name=path.name,
                sha256=_sha256_file(path),
            )
            for path in combined_images
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
    task_type: str,
    control_bundle_path: Optional[Path],
    motion_reference_video: Optional[Path],
    driving_audio: Optional[Path],
    output_dir: Path,
    quality: str,
    aspect_ratio: str,
    job_config: dict[str, object],
    progress: Optional[ProgressCallback] = None,
) -> Path:
    motion_reference_profile = analyze_motion_reference_video(motion_reference_video, output_dir)
    control_bundle = load_control_bundle(control_bundle_path) if control_bundle_path else None
    if progress:
        metadata = {
            "task_type": task_type,
            "motion_preset": job_config.get("motion_preset"),
            "style_preset": job_config.get("style_preset"),
            "duration_seconds": job_config.get("duration_seconds"),
        }
        if control_bundle:
            metadata["control_frames"] = control_bundle.sampled_frames
        if motion_reference_profile:
            metadata["motion_type"] = motion_reference_profile.get("motion_type")
            metadata["motion_summary"] = motion_reference_profile.get("motion_summary")
        progress(
            "shot_plan",
            metadata,
        )

    shot_plan_path = output_dir / "shot_plan.json"
    fps = 24 if quality == "max" else 20
    plan = ShotPlan(
        version=CONTRACT_VERSION,
        task_type=task_type,
        identity_pack_path=str(identity_pack_path),
        control_bundle_path=(
            str(control_bundle_path) if control_bundle_path and control_bundle_path.exists() else None
        ),
        prompt=str(job_config.get("prompt") or ""),
        negative_prompt=str(job_config["negative_prompt"]) if job_config.get("negative_prompt") else None,
        motion_preset=str(job_config["motion_preset"]) if job_config.get("motion_preset") else None,
        style_preset=str(job_config["style_preset"]) if job_config.get("style_preset") else None,
        duration_seconds=int(job_config.get("duration_seconds", 5)),
        seed=int(job_config["seed"]) if job_config.get("seed") is not None else None,
        motion_reference_video=(
            str(motion_reference_video) if motion_reference_video and motion_reference_video.exists() else None
        ),
        motion_reference_profile=motion_reference_profile,
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

    shot_plan = load_shot_plan(shot_plan_path)
    if shot_plan.task_type == "portrait_reenactment":
        render_command = os.getenv("PORTRAIT_REENACTMENT_RENDER_COMMAND", "").strip() or (
            "python3 /worker/src/generation_render_reenactment.py"
        )
    elif shot_plan.task_type == "full_body_reenactment":
        render_command = os.getenv("FULL_BODY_REENACTMENT_RENDER_COMMAND", "").strip() or (
            "python3 /worker/src/generation_render_full_body_reenactment.py"
        )
    else:
        render_command = os.getenv("GENERATION_RENDER_COMMAND", "").strip() or os.getenv(
            "GENERATION_PIPELINE_COMMAND", ""
        ).strip() or "python3 /worker/src/generation_render_cogvideox.py"
    report_path = output_video.with_suffix(".render-report.json")
    if render_command:
        cmd = [*shlex.split(render_command), "--shot-plan", str(shot_plan_path), "--output", str(output_video)]
        if shot_plan.identity_pack_path:
            cmd.extend(["--identity-pack", str(shot_plan.identity_pack_path)])
        if shot_plan.control_bundle_path:
            cmd.extend(["--control-bundle", str(shot_plan.control_bundle_path)])
        cmd.extend(["--report", str(report_path)])
        try:
            _run(cmd)
        except GenerationError as exc:
            if _report_flag_unsupported(str(exc)):
                fallback_cmd = [*shlex.split(render_command), "--shot-plan", str(shot_plan_path), "--output", str(output_video)]
                if shot_plan.identity_pack_path:
                    fallback_cmd.extend(["--identity-pack", str(shot_plan.identity_pack_path)])
                if shot_plan.control_bundle_path:
                    fallback_cmd.extend(["--control-bundle", str(shot_plan.control_bundle_path)])
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
            if _report_flag_unsupported(str(exc)):
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
