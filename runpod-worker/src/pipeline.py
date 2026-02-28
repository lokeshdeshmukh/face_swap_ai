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
VALID_SELECTOR_MODES = {"one", "many", "reference"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _is_asset_validation_error(message: str) -> bool:
    text = message.lower()
    return "validating source for" in text or "deleting corrupt source" in text


def _normalize_pixel_boost(value: str) -> str:
    if value == "256x256":
        return "512x512"
    if value not in VALID_FACE_SWAPPER_PIXEL_BOOST:
        logger.warning("invalid pixel boost=%s; falling back to 512x512", value)
        return "512x512"
    return value


def _dedupe_preserve_order(values: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


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


def _restore_audio_from_driving_if_missing(driving_audio: Path, output_video: Path) -> None:
    if not driving_audio.exists() or not output_video.exists():
        return
    if not _has_audio_stream(driving_audio):
        logger.info("driving audio has no audio stream; skipping audio restore")
        return
    if _has_audio_stream(output_video):
        logger.info("photo_sing output already contains audio stream")
        return

    remuxed = output_video.with_name(f"{output_video.stem}.with-audio{output_video.suffix}")
    remux_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(output_video),
        "-i",
        str(driving_audio),
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
    logger.info("restored audio stream from driving audio into photo_sing output")


def run_video_swap(
    source_images: List[Path],
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
    if not source_images:
        raise PipelineError("at least one source image is required for video_swap")

    quality_key = (quality or "balanced").strip().lower()
    is_max_quality = quality_key in {"max", "high"}

    def _profile_env(base_name: str, default: str) -> str:
        quality_name = f"{base_name}_{quality_key.upper()}"
        return os.getenv(quality_name, os.getenv(base_name, default))

    execution_provider = os.getenv("FACEFUSION_EXECUTION_PROVIDER", "cuda")
    download_providers = _dedupe_preserve_order(
        os.getenv("FACEFUSION_DOWNLOAD_PROVIDERS", "huggingface github").split()
    )
    if not download_providers:
        download_providers = ["huggingface", "github"]
    preferred_download_provider = download_providers[0]

    face_selector_mode = os.getenv("FACEFUSION_FACE_SELECTOR_MODE", "one")
    if face_selector_mode not in VALID_SELECTOR_MODES:
        logger.warning("invalid FACEFUSION_FACE_SELECTOR_MODE=%s; falling back to one", face_selector_mode)
        face_selector_mode = "one"
    adaptive_selector_enabled = _env_bool("FACEFUSION_ADAPTIVE_SELECTOR", True)
    try:
        probe_frames = int(os.getenv("FACEFUSION_PROBE_FRAMES", "72"))
    except ValueError:
        probe_frames = 72
    if probe_frames < 1:
        adaptive_selector_enabled = False

    selector_candidates = _dedupe_preserve_order(
        os.getenv(
            "FACEFUSION_SELECTOR_CANDIDATES",
            f"{face_selector_mode} reference many",
        ).split()
    )
    selector_candidates = [mode for mode in selector_candidates if mode in VALID_SELECTOR_MODES]
    if not selector_candidates:
        selector_candidates = [face_selector_mode]

    face_selector_order = os.getenv("FACEFUSION_FACE_SELECTOR_ORDER", "large-small")
    reference_face_position = os.getenv("FACEFUSION_REFERENCE_FACE_POSITION", "0")
    reference_frame_number = os.getenv("FACEFUSION_REFERENCE_FRAME_NUMBER", "0")
    reference_face_distance = os.getenv("FACEFUSION_REFERENCE_FACE_DISTANCE", "0.6")
    face_detector_model = os.getenv("FACEFUSION_FACE_DETECTOR_MODEL", "yolo_face")
    # Accept common alias and normalize to FaceFusion CLI value.
    if face_detector_model == "yoloface":
        face_detector_model = "yolo_face"
    face_detector_size = os.getenv("FACEFUSION_FACE_DETECTOR_SIZE", "640x640")
    face_detector_score = _profile_env("FACEFUSION_FACE_DETECTOR_SCORE", "0.20" if is_max_quality else "0.30")
    face_landmarker_score = _profile_env("FACEFUSION_FACE_LANDMARKER_SCORE", "0.20" if is_max_quality else "0.30")
    face_detector_angles = os.getenv("FACEFUSION_FACE_DETECTOR_ANGLES", "0 90 180 270").split()
    face_swapper_weight = _profile_env("FACEFUSION_FACE_SWAPPER_WEIGHT", "1.00" if is_max_quality else "0.85")
    face_swapper_pixel_boost = _normalize_pixel_boost(
        _profile_env("FACEFUSION_FACE_SWAPPER_PIXEL_BOOST", "1024x1024" if is_max_quality else "768x768")
    )
    output_video_encoder = _profile_env("FACEFUSION_OUTPUT_VIDEO_ENCODER", "h264_nvenc")
    log_level = os.getenv("FACEFUSION_LOG_LEVEL", "info")

    default_model = "simswap_unofficial_512" if is_max_quality else "inswapper_128_fp16"
    model = _profile_env("FACEFUSION_MODEL", default_model)

    common_cmd = [
        "python3",
        "facefusion.py",
        "headless-run",
        "--processors",
        "face_swapper",
        "-s",
        *[str(path) for path in source_images],
        "-t",
        str(target_video),
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
        "--output-video-encoder",
        output_video_encoder,
        "--log-level",
        log_level,
    ]

    def _run_swap_with_mode(
        selector_mode: str,
        provider_values: List[str],
        destination: Path,
        trim_end_frame: Optional[int] = None,
    ) -> str:
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
            *common_cmd,
            "-o",
            str(destination),
            *selector_args,
            "--download-providers",
            *provider_values,
            "--face-swapper-model",
            model,
            "--execution-providers",
            execution_provider,
        ]
        if trim_end_frame is not None:
            cmd.extend(
                [
                    "--trim-frame-start",
                    "0",
                    "--trim-frame-end",
                    str(trim_end_frame),
                ]
            )
        return _run(cmd, cwd=facefusion_root)

    def _select_mode(provider_values: List[str]) -> str:
        if not adaptive_selector_enabled:
            return face_selector_mode
        probe_output = output_video.with_name(f"{output_video.stem}.probe.mp4")
        try:
            for candidate_mode in selector_candidates:
                try:
                    probe_log = _run_swap_with_mode(
                        candidate_mode,
                        provider_values,
                        probe_output,
                        trim_end_frame=probe_frames,
                    )
                except PipelineError as exc:
                    if _is_asset_validation_error(str(exc)):
                        raise
                    logger.warning("adaptive probe failed for selector_mode=%s: %s", candidate_mode, exc)
                    continue
                if _looks_like_no_face_detected(probe_log):
                    logger.info("adaptive probe selector_mode=%s indicated no-face", candidate_mode)
                    continue
                logger.info("adaptive probe selected selector_mode=%s", candidate_mode)
                return candidate_mode
        finally:
            if probe_output.exists():
                probe_output.unlink(missing_ok=True)
        logger.warning("adaptive probe found no confident selector mode; using default=%s", face_selector_mode)
        return face_selector_mode

    def _run_full(provider_values: List[str]) -> None:
        selected_mode = _select_mode(provider_values)
        full_output = _run_swap_with_mode(selected_mode, provider_values, output_video)
        if _looks_like_no_face_detected(full_output) and selected_mode != "many":
            logger.warning("full run mode=%s indicated no-face; retrying with selector_mode=many", selected_mode)
            _run_swap_with_mode("many", provider_values, output_video)

    try:
        _run_full(download_providers)
    except PipelineError as exc:
        if not _is_asset_validation_error(str(exc)):
            raise
        alternate_provider = "github" if preferred_download_provider == "huggingface" else "huggingface"
        _run_full([alternate_provider])

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
    try:
        _restore_audio_from_driving_if_missing(driving_audio, output_video)
    except Exception as exc:
        logger.warning("photo_sing audio restore step failed: %s", exc)


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
