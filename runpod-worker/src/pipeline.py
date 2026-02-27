from __future__ import annotations

import shutil
import subprocess
import os
import logging
from pathlib import Path
from typing import List, Optional


class PipelineError(RuntimeError):
    pass


logger = logging.getLogger("runpod-worker")
VALID_FACE_SWAPPER_PIXEL_BOOST = {"512x512", "768x768", "1024x1024"}


def _run(cmd: List[str], cwd: Optional[Path] = None) -> str:
    logger.info("running command: %s", " ".join(cmd))
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    lines: List[str] = []
    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.rstrip()
        lines.append(line)
        logger.info("[cmd] %s", line)

    return_code = process.wait()
    output = "\n".join(lines)
    if return_code != 0:
        raise PipelineError(
            f"command failed: {' '.join(cmd)}\nstdout+stderr:\n{output}"
        )
    return output


def _looks_like_no_face_detected(output: str) -> bool:
    text = output.lower()
    signals = [
        "no source face",
        "no target face",
        "no face found",
        "no faces found",
        "could not detect a face",
        "could not detect faces",
    ]
    return any(signal in text for signal in signals)


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
        logger.warning("ffprobe failed for %s: %s", video_path, result.stderr.strip())
        return False
    return "audio" in result.stdout.lower()


def _restore_audio_if_missing(target_video: Path, output_video: Path) -> None:
    if not target_video.exists() or not output_video.exists():
        return
    if not _has_audio_stream(target_video):
        logger.info("target video has no audio stream; skipping audio restore")
        return
    if _has_audio_stream(output_video):
        logger.info("output already contains audio stream")
        return

    remuxed = output_video.with_name(f"{output_video.stem}.with-audio{output_video.suffix}")
    remux_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(output_video),
        "-i",
        str(target_video),
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
    _run(remux_cmd)
    remuxed.replace(output_video)
    logger.info("restored audio stream from target video into output")


def run_video_swap(
    source_image: Path,
    target_video: Path,
    output_video: Path,
    quality: str,
) -> None:
    """
    Requires a FaceFusion installation available in PATH.
    The command is an example template and may need adaptation to your exact image.
    """
    facefusion_root = Path("/opt/facefusion")
    facefusion_entry = facefusion_root / "facefusion.py"
    if not facefusion_entry.exists():
        raise PipelineError("facefusion entrypoint not found in worker image")

    execution_provider = os.getenv("FACEFUSION_EXECUTION_PROVIDER", "cuda")
    download_providers = os.getenv("FACEFUSION_DOWNLOAD_PROVIDERS", "huggingface github").split()
    if not download_providers:
        download_providers = ["huggingface", "github"]
    preferred_download_provider = download_providers[0]
    face_selector_mode = os.getenv("FACEFUSION_FACE_SELECTOR_MODE", "one")
    face_selector_order = os.getenv("FACEFUSION_FACE_SELECTOR_ORDER", "large-small")
    reference_face_position = os.getenv("FACEFUSION_REFERENCE_FACE_POSITION", "0")
    reference_frame_number = os.getenv("FACEFUSION_REFERENCE_FRAME_NUMBER", "0")
    reference_face_distance = os.getenv("FACEFUSION_REFERENCE_FACE_DISTANCE", "0.6")
    face_detector_model = os.getenv("FACEFUSION_FACE_DETECTOR_MODEL", "yolo_face")
    # Accept common alias and normalize to FaceFusion CLI value.
    if face_detector_model == "yoloface":
        face_detector_model = "yolo_face"
    face_detector_size = os.getenv("FACEFUSION_FACE_DETECTOR_SIZE", "640x640")
    face_detector_score = os.getenv("FACEFUSION_FACE_DETECTOR_SCORE", "0.35")
    face_landmarker_score = os.getenv("FACEFUSION_FACE_LANDMARKER_SCORE", "0.35")
    face_detector_angles = os.getenv("FACEFUSION_FACE_DETECTOR_ANGLES", "0 90 180 270").split()
    face_swapper_weight = os.getenv("FACEFUSION_FACE_SWAPPER_WEIGHT", "1.0")
    face_swapper_pixel_boost = os.getenv("FACEFUSION_FACE_SWAPPER_PIXEL_BOOST", "512x512")
    if face_swapper_pixel_boost == "256x256":
        face_swapper_pixel_boost = "512x512"
    if face_swapper_pixel_boost not in VALID_FACE_SWAPPER_PIXEL_BOOST:
        logger.warning(
            "invalid FACEFUSION_FACE_SWAPPER_PIXEL_BOOST=%s; falling back to 512x512",
            face_swapper_pixel_boost,
        )
        face_swapper_pixel_boost = "512x512"
    log_level = os.getenv("FACEFUSION_LOG_LEVEL", "info")

    model = "inswapper_128"
    if quality == "max":
        model = "simswap_unofficial_512"

    base_cmd = [
        "python3",
        "facefusion.py",
        "headless-run",
        "--processors",
        "face_swapper",
        "-s",
        str(source_image),
        "-t",
        str(target_video),
        "-o",
        str(output_video),
        "--face-detector-model",
        face_detector_model,
        "--face-detector-size",
        face_detector_size,
        "--face-detector-score",
        face_detector_score,
        "--face-landmarker-score",
        face_landmarker_score,
        "--face-detector-angles",
        *face_detector_angles,
        "--face-swapper-weight",
        face_swapper_weight,
        "--face-swapper-pixel-boost",
        face_swapper_pixel_boost,
        "--log-level",
        log_level,
    ]

    def _run_swap_with_mode(selector_mode: str, provider_values: List[str]) -> str:
        selector_args = [
            "--face-selector-mode",
            selector_mode,
            "--face-selector-order",
            face_selector_order,
        ]
        if selector_mode == "reference":
            selector_args.extend(
                [
                    "--reference-face-position",
                    reference_face_position,
                    "--reference-frame-number",
                    reference_frame_number,
                    "--reference-face-distance",
                    reference_face_distance,
                ]
            )
        cmd = [
            *base_cmd,
            *selector_args,
            "--download-providers",
            *provider_values,
            "--face-swapper-model",
            model,
            "--execution-providers",
            execution_provider,
        ]
        return _run(cmd, cwd=facefusion_root)

    try:
        first_output = _run_swap_with_mode(face_selector_mode, download_providers)
        if _looks_like_no_face_detected(first_output) and face_selector_mode != "many":
            _run_swap_with_mode("many", download_providers)
    except PipelineError as exc:
        # Retry once with alternate provider when an asset hash/source validation fails.
        message = str(exc).lower()
        if "validating source for" not in message and "deleting corrupt source" not in message:
            raise
        alternate_provider = "github" if preferred_download_provider == "huggingface" else "huggingface"
        retry_providers = [alternate_provider]
        retry_output = _run_swap_with_mode(face_selector_mode, retry_providers)
        if _looks_like_no_face_detected(retry_output) and face_selector_mode != "many":
            _run_swap_with_mode("many", retry_providers)

    try:
        _restore_audio_if_missing(target_video, output_video)
    except Exception as exc:
        logger.warning("audio restore step failed: %s", exc)


def run_photo_sing(
    source_image: Path,
    driving_video: Path,
    driving_audio: Path,
    output_video: Path,
) -> None:
    """
    Template pipeline:
    1) LivePortrait to drive image with influencer video motion.
    2) MuseTalk to tighten lip sync against provided audio.

    This assumes both repos are pre-installed in the worker image with CLI entry points.
    """
    if shutil.which("liveportrait") is None:
        raise PipelineError("liveportrait executable not found in worker image")
    if shutil.which("musetalk") is None:
        raise PipelineError("musetalk executable not found in worker image")

    animated = output_video.with_name("animated.mp4")

    liveportrait_cmd = [
        "liveportrait",
        "--source-image",
        str(source_image),
        "--driving-video",
        str(driving_video),
        "--output",
        str(animated),
    ]
    _run(liveportrait_cmd)

    musetalk_cmd = [
        "musetalk",
        "--video",
        str(animated),
        "--audio",
        str(driving_audio),
        "--output",
        str(output_video),
    ]
    _run(musetalk_cmd)


def run_4k_enhance(input_video: Path, output_video: Path) -> None:
    """
    Optional enhancement pass.
    Requires realesrgan-ncnn-vulkan or equivalent binary in worker image.
    """
    if shutil.which("realesrgan-ncnn-vulkan") is None:
        shutil.copy2(input_video, output_video)
        return

    cmd = [
        "realesrgan-ncnn-vulkan",
        "-i",
        str(input_video),
        "-o",
        str(output_video),
        "-s",
        "2",
    ]
    _run(cmd)
