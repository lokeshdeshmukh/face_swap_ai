from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class PipelineError(RuntimeError):
    pass


def _run(cmd: list[str], cwd: Path | None = None) -> None:
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
    if shutil.which("facefusion") is None:
        raise PipelineError("facefusion executable not found in worker image")

    execution_provider = "cuda"
    model = "inswapper_128"
    if quality == "max":
        model = "inswapper-512-live"

    cmd = [
        "facefusion",
        "headless-run",
        "--source",
        str(source_image),
        "--target",
        str(target_video),
        "--output-path",
        str(output_video),
        "--face-swapper-model",
        model,
        "--execution-providers",
        execution_provider,
    ]
    _run(cmd)


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
