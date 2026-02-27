from __future__ import annotations

import shutil
import subprocess
import os
from pathlib import Path
from typing import List, Optional


class PipelineError(RuntimeError):
    pass


def _run(cmd: List[str], cwd: Optional[Path] = None) -> None:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise PipelineError(
            f"command failed: {' '.join(cmd)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


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
    preferred_download_provider = os.getenv("FACEFUSION_DOWNLOAD_PROVIDER", "huggingface")
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
    ]

    cmd = [
        *base_cmd,
        "--download-providers",
        preferred_download_provider,
        "--face-swapper-model",
        model,
        "--execution-providers",
        execution_provider,
    ]
    try:
        _run(cmd, cwd=facefusion_root)
    except PipelineError as exc:
        # Retry once with alternate provider when an asset hash/source validation fails.
        message = str(exc).lower()
        if "validating source for" not in message and "deleting corrupt source" not in message:
            raise
        alternate_provider = "github" if preferred_download_provider == "huggingface" else "huggingface"
        retry_cmd = [
            *base_cmd,
            "--download-providers",
            alternate_provider,
            "--face-swapper-model",
            model,
            "--execution-providers",
            execution_provider,
        ]
        _run(retry_cmd, cwd=facefusion_root)


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
