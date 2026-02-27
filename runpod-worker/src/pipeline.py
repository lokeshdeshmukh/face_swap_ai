from __future__ import annotations

import shutil
import subprocess
import os
from pathlib import Path
from typing import List, Optional


class PipelineError(RuntimeError):
    pass


def _run(cmd: List[str], cwd: Optional[Path] = None) -> str:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    output = f"{result.stdout}\n{result.stderr}"
    if result.returncode != 0:
        raise PipelineError(
            f"command failed: {' '.join(cmd)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
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
    face_swapper_pixel_boost = os.getenv("FACEFUSION_FACE_SWAPPER_PIXEL_BOOST", "256x256")
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
